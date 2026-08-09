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

Doubling to a 128x128 tile doubles intensity to 102 FLOP/byte, predicting
~143 TFLOP/s. `BK` dropped to 32 to keep both double-buffered tiles inside the
48 KB shared-memory budget, which costs nothing since `BK` cancels out above.

| tile | predicted ceiling | measured (4096x4096, M=4096) |
|---|---|---|
| 64x64 | ~71 TFLOP/s | 73.8 TFLOP/s |
| 128x128 | ~143 TFLOP/s | **100.1 TFLOP/s** |

The prediction held for the small tile and was optimistic for the large one:
1.36x gained against 1.95x predicted, landing at 70% of the new ceiling. `ncu`
says why.

### Why the prefill kernel stops at 70% of its roofline

```
launch__registers_per_thread                    128
sm__warps_active (achieved occupancy)         24.5 %
sm__inst_executed_pipe_tensor.sum         33,554,432
smsp__inst_executed.sum                  319,463,424
```

**About 9.5 non-tensor instructions issue for every tensor instruction.** The
kernel is issue-bound on overhead — address arithmetic, shared-memory loads,
loop bookkeeping — not on math. 128 registers per thread at 256 threads means
exactly 2 blocks per SM, which is where the 24.5% occupancy comes from.

The cause is the wmma API. `load_matrix_sync` recomputes fragment addressing on
every call and gives no control over shared-memory swizzling, so a large share
of the instruction stream is overhead wmma will not let the kernel remove. That
is the acknowledged cost of using it: correct by construction, with a ceiling.

**Honest framing of the prefill numbers.** The fused kernel reaches 39% of
cuBLAS fp16. Dequantising to fp16 and calling cuBLAS reaches 99%. At prefill,
dequant-then-cuBLAS is the better engineering choice on time alone, and the
fused kernel earns its place only by never materialising the fp16 matrix — it
saves the memory, which is the reason INT4 exists. `ops.matmul` dispatches on
batch size for the decode win and does not pretend the prefill kernel is
something it is not.

---

## What would come next

1. **Raw `mma.sync` + `ldmatrix` for the prefill kernel.** The 9.5:1 instruction
   ratio is the target; precomputed fragment addressing and an XOR-swizzled
   shared layout would remove most of the non-tensor instructions wmma forces.
2. **Fuse the split-K reduction.** A cooperative-groups grid sync or a
   deterministic atomic reduction would remove one of the two launches, worth
   ~3 us — around 20% on the smallest decode shapes.
3. **`TN` as a tunable.** It is the one decode knob shown to change available
   parallelism that is still hard-coded at 4.
4. **Shared-memory swizzling instead of padding.** `LDB = BK + 8` leaves a
   4-way conflict on the B fragment path; an XOR swizzle removes it without the
   padding overhead.
5. **Sub-group split-K.** Splits are capped at `K / group_size`; splitting
   inside a group would lift the cap that limits the narrow shapes.
