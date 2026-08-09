# Optimisation log

Every number here was measured on one NVIDIA A100-SXM4-40GB (sm_80, 108 SMs,
40 MB L2), CUDA 12.8, PyTorch 2.11, group size 128, fp16 activations. Raw CSVs
are in `docs/results/`; the scripts that produced them are in `bench/`.

Two numbers anchor everything:

* **Measured achievable bandwidth: ~1375 GB/s.** From a large device-to-device
  copy on this machine, not the 1555 GB/s on the spec sheet. Every "% of peak"
  below is against the measured figure.
* **Bandwidth ceiling vs fp16: 3.88x.** At group size 128 a 4-bit weight matrix
  plus its scales is 3.88x smaller than the fp16 equivalent. In the
  bandwidth-bound regime that is the *most* any INT4 kernel can beat cuBLAS
  fp16 by. A kernel claiming more than 3.88x is measuring its cache, not its
  kernel.

---

## The measurement itself

Two decisions in `bench/harness.py` matter more than any kernel change.

**L2 defeat.** A 4096x4096 INT4 weight matrix is 8 MB. The A100's L2 is 40 MB.
Time the same buffer in a loop and everything after the first iteration is
served from L2 at several TB/s. That does not just inflate the result, it
inflates it *asymmetrically*: 8 MB of INT4 weights fit in L2 and the 32 MB fp16
baseline does not, so the comparison invents a speedup out of nothing. The
harness allocates enough distinct weight copies to overflow L2 and rotates
through them, which is also what a real forward pass does — every layer is
touched once and never revisited.

**Peak from measurement, not the datasheet.** Quoting 1555 GB/s would make every
kernel here look ~12% worse than it is against a number nothing can reach.

---

## Decode regime (small M): the ladder

Bandwidth-bound. Headline metric is GB/s, not TFLOP/s. Shape 4096x14336
(a Llama-3-8B gate/up projection), M=1.

| stage | what changed | ms | GB/s | vs previous |
|---|---|---|---|---|
| v0 naive | thread per output, scalar decode | 0.534 | 57 | — |
| v1 coalesced | reuse each packed word for all 8 nibbles | 0.297 | 102 | 1.80x |
| v2 fast decode | `lop3` bit-trick decode, half2 FMA, smem activations | 0.073 | 417 | 4.09x |
| v3 vectorised | 128-bit loads, 4 output columns per thread | 0.132 | 229 | **0.55x** |
| v4 split-K | split the reduction axis across blocks | 0.031 | 970 | 4.23x |

For reference on the same shape: cuBLAS fp16 runs in 0.097 ms, a hand-written
Triton W4A16 kernel in 0.137 ms, and dequantise-then-cuBLAS in 0.225 ms.

**Net 17.1x over the naive kernel, 3.11x over cuBLAS fp16, and 4.4x over the
Triton baseline** — 80% of the 3.88x ceiling the byte ratio allows.

Across all four shapes at M=1, v4 lands at 2.1x–3.1x over cuBLAS fp16
(55%–80% of ceiling); the weakest is 4096x4096, for reasons diagnosed below.

### v1: the load, not the math

The naive kernel re-reads the same packed `uint32` once per nibble. Using each
load for all 8 values it carries is worth 1.8x before touching arithmetic.

### v2: making dequantisation nearly free

This is the largest single win in the file (4.1x) and the least obvious.

The direct way to turn a 4-bit integer into fp16 is shift, mask, integer-to-float
convert, multiply. The I2F is the problem: at one per weight it saturates the
conversion pipe long before HBM saturates, so a kernel whose entire purpose is
to be bandwidth-bound ends up arithmetic-bound instead.

The fix exploits the fp16 bit layout. `0x6400` is exactly `1024.0` with a zero
mantissa, and at that exponent the low mantissa bits are integer-valued — so
OR-ing a nibble `n` into `0x6400` produces exactly the float `1024 + n`, with no
conversion instruction. Recovering `n` is one subtract, and since two fp16 fit
in a register that subtract is a packed `hsub2` doing two weights at once.
Extraction is a single `lop3.b32` (one instruction for `(a & b) | c`) with masks
that pick up two nibbles 16 bits apart, so results land pre-packed as `half2`.

Eight weights cost 4 `lop3` + 2 `hsub2` + 2 `hfma2`, against roughly 8 shifts +
8 masks + 8 I2F + 8 multiplies. The lane masked at bits 4..7 comes out 16x too
large; instead of shifting, an `hfma2` folds the 1/16 and the bias together —
and the quantiser's implicit zero-point of 8 folds into that same bias, making
zero-point correction free.

**The decode is bit-exact.** `tests/test_dequant.py` sweeps all 16 nibble values
in all 8 slots and requires `torch.equal` against both a scalar CUDA decode and
a NumPy reference. This matters more than the speed: a wrong magic constant here
produces plausible-looking output and a silently degraded model, which an
end-to-end tolerance check would happily pass.

### v3: a change that made things worse

Switching to 128-bit loads with 4 output columns per thread — textbook
vectorisation — **cost 45% of performance**.

The reason is launch geometry, not memory. Giving each thread 4 columns means
each block covers 512 columns, so a 4096-column layer produces 8 blocks. On 108
SMs, more than 90% of the GPU is idle. The vectorisation was fine; it just
bought nothing because the kernel was never issue-limited, and it made the
parallelism problem worse.

This rung is kept in the ladder rather than quietly deleted because it is the
whole reason v4 exists.

### v4: split-K

Splitting the reduction axis across blocks and summing fp32 partials in a second
pass restores the parallelism v3 gave up. Measured in isolation by
`bench/autotune.py`, split-K alone is worth **4.6x to 13x** depending on shape:

| shape | splits=1 | best splits | speedup |
|---|---|---|---|
| 4096x4096 | 0.132 ms | 0.014 ms (32) | 9.2x |
| 4096x14336 | 0.143 ms | 0.031 ms (16) | 4.6x |
| 14336x4096 | 0.418 ms | 0.032 ms (48) | 13.1x |
| 8192x8192 | 0.251 ms | 0.036 ms (32) | 7.0x |

Partials go to a `[splits, M, N]` fp32 workspace summed by a second kernel —
deterministic, unlike `atomicAdd`, and a few hundred KB. `partial == nullptr`
selects a single-split path that writes fp16 directly and skips the second
launch entirely.

**A rounding bug worth naming.** Converting a split count into groups-per-split
with `ceil()` and back silently halves the grid: 32 groups split 27 ways gives 2
groups each, which is only 16 splits. Flooring instead was worth 8–18% on three
of four shapes, for a one-character change.

### A second negative result: block width does nothing

Having found that v3 was parallelism-starved, the obvious next move was to
narrow the block — more blocks, more loads in flight. The sweep says otherwise:
32, 64 and 128 threads per block land within 0.1% of each other.

The hypothesis was simply wrong, and the arithmetic says so. Each thread owns
`TN=4` output columns, so the total thread count is `(N / TN) * splits`
regardless of how those threads are grouped into blocks. Narrowing the block
makes more, smaller blocks out of exactly the same warps. The knobs that
actually add memory-level parallelism are the split count (capped at
`K / group_size`) and `TN`.

The override survives in the API so the sweep can keep re-proving this on other
hardware, but the default is now a constant.

### Where the remaining gap goes

At 4096x4096, M=1, the kernel reaches 525 GB/s — well below the ~970 GB/s the
larger shapes hit. Capturing the same calls into a CUDA graph answers why:

| shape | eager | CUDA graph | launch cost | graph GB/s | % peak |
|---|---|---|---|---|---|
| 4096x4096 | 0.0165 ms | 0.0130 ms | 3.6 us | 669 | 48.5% |
| 4096x14336 | 0.0327 ms | 0.0299 ms | 2.8 us | 1014 | **73.5%** |
| 14336x4096 | 0.0350 ms | 0.0318 ms | 3.2 us | 953 | 69.1% |
| 8192x8192 | 0.0379 ms | 0.0348 ms | 3.1 us | 995 | 72.2% |

Launch overhead is a flat ~3 us. On a layer that takes 33 us that is noise; on
one that takes 16.5 us it is 21% of wall-clock. **The smallest layers are
latency-bound, not bandwidth-bound** — no amount of memory tuning would have
moved them, and the fix is structural (fewer launches, or graph capture, which
is how a serving loop runs anyway).

`ncu` corroborates: the decode kernel runs at 14.5% achieved occupancy and 60
registers per thread. For a streaming kernel that is not a defect — it needs
loads in flight, not resident warps — and the split ceiling of `K / group_size`
caps how many warps the shape can even provide.

---

## Prefill regime (large M): arithmetic intensity decides

Once M is large the weights are reused across rows, HBM stops being the limit,
and the constraint becomes how many FLOP the kernel extracts per byte staged
into shared memory. For a `BM x BN` tile stepping `BK`:

```
FLOP  per tile = 2 * BM * BN * BK
bytes per tile = BM * BK * 2   (fp16 activations) + BN * BK / 2  (INT4 weights)
```

`BK` cancels. **The tile size alone sets the ceiling.**

A 64x64 tile gives 51 FLOP/byte. The A100 needs roughly
`312 TFLOP/s / 1.4 TB/s ≈ 220` FLOP/byte to saturate its tensor cores, so a
64x64 tile is memory-bound by construction and caps near
`51 * 1.4 TB/s ≈ 71 TFLOP/s`.

The first version measured **73.8 TFLOP/s** — its roofline, not a defect. No
inner-loop tuning would have moved it.

What follows is four hypotheses about the remaining gap, in the order they were
tested. Two were right. Shape 4096x4096, M=4096.

| # | hypothesis | outcome | TFLOP/s |
|---|---|---|---|
| — | starting point (64x64 tile) | — | 73.8 |
| 1 | the tile is too small — intensity caps it | **confirmed** | 100.1 |
| 2 | wmma's addressing overhead is the limit | **disproved** | 100.1 |
| 3 | a runtime integer division is on the hot path | **confirmed** | 111.7 |
| 4 | 64-bit address arithmetic still costs | instructions down 9%, **no speedup** | 111.7 |

### 1. Tile size — confirmed

Doubling to 128x128 doubles intensity to 102 FLOP/byte, predicting
~143 TFLOP/s. `BK` dropped to 32 to keep both double-buffered tiles inside the
48 KB shared-memory budget, which costs nothing since `BK` cancels above.

Measured 100.1 TFLOP/s: the direction was right, the magnitude optimistic —
1.36x gained against 1.95x predicted, landing at 70% of the new ceiling. `ncu`
was the obvious next step:

```
launch__registers_per_thread                    128
sm__warps_active (achieved occupancy)         24.5 %
sm__inst_executed_pipe_tensor.sum         33,554,432
smsp__inst_executed.sum                  319,463,424
```

**9.5 non-tensor instructions per tensor instruction.** The kernel was issue-bound
on overhead, not math.

### 2. wmma is the overhead — disproved

The natural reading of that ratio is that `load_matrix_sync` recomputes fragment
addressing on every call, so the fix is to drop to raw `mma.sync` + `ldmatrix`,
hoist all fragment addresses out of the k-loop, and let the accumulator layout
write straight to global memory (d0/d1 land in adjacent columns, so each pair is
one `half2` store — no shared-memory staging epilogue at all).

That kernel is in `csrc/gemm_w4a16.cu` as v7 and it is **bit-identical** to the
wmma version, which is a satisfying confirmation that the hand-derived fragment
layouts in `csrc/mma.cuh` are right.

It is also **not faster**. 1.26 ms against wmma's 1.23 ms, with *more*
instructions (332.6M vs 319.5M), presumably because hoisting addresses into
registers costs moves the compiler was avoiding. The hypothesis was wrong: nvcc
already lowers wmma to essentially the same ldmatrix/mma sequence, and wmma was
never the overhead.

Both kernels are kept, and v6 is the default because it is marginally the
faster of the two. The value of v7 is that it makes the cost of the wmma
abstraction a *measured* number — roughly zero — rather than an assumption.

### 3. A runtime integer division — confirmed

With wmma exonerated, the question became where the instructions actually go.
Rather than guess a third time, the right move was a pipe breakdown:

```
smsp__inst_executed_pipe_fma.sum         131,432,448   (39.5%)
smsp__inst_executed_pipe_alu.sum          82,960,384   (25.0%)
smsp__inst_executed_pipe_lsu.sum          25,722,880    (7.7%)
smsp__inst_executed_pipe_xu.sum            6,291,456    (1.9%)
sm__inst_executed_pipe_tensor.sum         33,554,432   (10.1%)
```

On NVIDIA GPUs integer multiply-add issues on the **FMA pipe**, so 64% of this
kernel on FMA+ALU means address arithmetic, not math. And the XU pipe —
transcendentals and, crucially, integer division — had no business being
occupied at all.

The culprit was one line in the weight-staging loop:

```cpp
const half sc = S[((k0 + r8 * 8) / group_size) * N + n];
```

`group_size` was a kernel *argument*. Integer division by a runtime value
compiles to a multi-instruction sequence, executed once per thread per
iteration, for every weight tile in the kernel. Making it a template parameter
turns it into a shift — the decode kernel had templated it all along; the
prefill kernel had not.

**The XU pipe went to exactly zero**, total instructions fell 333M → 287M, and
throughput went 100.1 → 111.7 TFLOP/s.

### 4. 32-bit indexing — instructions down, clock unchanged

The address arithmetic was still widening to `size_t` before multiplying, making
every offset a 64-bit IMAD. Doing the arithmetic in 32 bits and widening only
the final offset (guarded by a host-side shape check) cut ALU another 36%:

```
smsp__inst_executed.sum        263,299,072   (was 332,595,200 — down 21%)
smsp__inst_executed_pipe_alu.sum 43,073,536   (was  82,960,384 — down 48%)
smsp__inst_executed_pipe_xu.sum           0
```

**Throughput did not move.** 111.7 TFLOP/s before and after.

That is the useful result: having removed 21% of the instruction stream for no
gain, the kernel is demonstrably **no longer issue-bound**. At 24% occupancy
with a two-stage pipeline it is latency-bound — not enough independent work in
flight to hide `mma` and shared-load latency. The next lever is a deeper (3–4
stage) `cp.async` pipeline and cutting the 124-register footprint, not more
instruction golf.

Final state of the shipped kernel (v6), for comparison with the 319.5M it
started at:

```
smsp__inst_executed.sum                  246,505,472   (down 23%)
smsp__inst_executed_pipe_xu.sum                    0
launch__registers_per_thread                     124
sm__warps_active                               23.9 %
```

### Honest framing of the prefill numbers

The fused kernel reaches 44% of cuBLAS fp16. Dequantising to fp16 and calling
cuBLAS reaches 99–101%. At prefill, dequant-then-cuBLAS is the better
engineering choice on time alone, and the fused kernel earns its place only by
never materialising the fp16 matrix — it saves the memory, which is the reason
INT4 exists. `ops.matmul` dispatches on batch size for the decode win and does
not pretend the prefill kernel is something it is not.

**The Triton baseline ties or wins here.** 111.7 TFLOP/s against this kernel's
111.5 at 4096x4096, and 114.5 against 105.9 at 4096x14336. That is the expected
shape of the result rather than an embarrassment: prefill is a well-trodden
tiled-GEMM problem where an autotuner sweeping block sizes, warps and stages is
hard to beat by hand in a few iterations. Decode is the opposite — an awkward,
launch-geometry-sensitive, bandwidth-bound shape where `tl.dot`'s 16-row minimum
tile wastes 15/16 of the work at M=1 — and there the CUDA kernel wins by 4.4x.
Which regime rewards hand-written CUDA, and which does not, is the more useful
finding than either number alone.

---

## What would come next

1. **Deeper prefill pipeline.** The kernel is latency-bound at 24% occupancy
   with two `cp.async` stages. Three or four stages, plus cutting the
   128-register footprint to fit a third block per SM, is the remaining lever —
   instruction count is not.
2. **Fuse the split-K reduction.** A cooperative-groups grid sync or a
   deterministic atomic reduction would remove one of the two launches, worth
   ~3 us — around 20% on the smallest decode shapes.
3. **`TN` as a tunable.** It is the one decode knob shown to change available
   parallelism that is still hard-coded at 4.
4. **Shared-memory swizzling instead of padding.** `LDB = BK + 8` happens to be
   conflict-free for `ldmatrix` (see `csrc/mma.cuh`) but costs 20% of the tile's
   shared footprint; an XOR swizzle would reclaim it.
5. **Sub-group split-K.** Splits are capped at `K / group_size`; splitting
   inside a group would lift the cap that limits the narrow shapes.
