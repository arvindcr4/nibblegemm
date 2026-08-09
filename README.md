# nibblegemm

INT4 weight-only (W4A16) GEMM kernels for NVIDIA Ampere, written in CUDA from
scratch and benchmarked honestly.

This is the kernel that decides how fast a quantised LLM generates tokens. At
batch 1 a decoder layer does almost no arithmetic per weight, so decode speed is
set by how fast weights can be pulled out of HBM — which is why every serving
stack (vLLM, SGLang, TensorRT-LLM) ships a hand-written W4A16 kernel rather than
calling cuBLAS.

**On an A100-SXM4-40GB, `nibblegemm` runs a Llama-3-8B MLP projection at batch 1
in 31 µs: 3.13x faster than cuBLAS fp16, 4.5x faster than a hand-written Triton
W4A16 kernel, and 81% of the theoretical ceiling that the 4-bit byte reduction
allows.**

---

## Results

A100-SXM4-40GB, CUDA 12.8, PyTorch 2.11, group size 128, M=1 (decode).
Measured achievable bandwidth on this machine is 1375 GB/s (a copy benchmark —
the spec sheet's 1555 GB/s is not reachable by anything).

| shape (K×N) | layer | cuBLAS fp16 | Triton W4A16 | **nibblegemm** | speedup | GB/s | % of ceiling |
|---|---|---|---|---|---|---|---|
| 4096×14336 | 8B gate/up | 0.097 ms | 0.138 ms | **0.031 ms** | **3.13x** | 977 | 81% |
| 14336×4096 | 8B down | 0.096 ms | 0.200 ms | **0.032 ms** | **2.98x** | 941 | 77% |
| 8192×8192 | 70B attn | 0.105 ms | 0.166 ms | **0.036 ms** | **2.93x** | 970 | 76% |
| 4096×4096 | 8B q/o | 0.031 ms | 0.077 ms | **0.014 ms** | **2.14x** | 603 | 55% |

**About "% of ceiling".** At group size 128, INT4 weights plus scales are 3.88x
smaller than fp16. In a bandwidth-bound regime that is the *most* any INT4
kernel can beat cuBLAS by — so 3.88x is the honest denominator, and anything
reporting a larger speedup is measuring its cache rather than its kernel. The
smallest shape reaching only 55% is a real limitation, diagnosed in the
[optimisation log](docs/OPTIMIZATION_LOG.md#where-the-remaining-gap-goes): at
16 µs, a flat ~3 µs of kernel-launch overhead is 21% of wall-clock. Under CUDA
graph capture (how a serving loop actually runs) the same kernels reach
**73.5% of measured peak bandwidth**.

Full data in [`docs/results/`](docs/results/); every table is reproducible with
the scripts in [`bench/`](bench/).

---

## Does it matter in a real model?

A fast kernel that does not move a decode step, or that ruins the model, is
worthless. Both are measured ([`bench/bench_model.py`](bench/bench_model.py)).

**Speed, at Llama-3-8B scale.** A decode step is ~224 projections, and the
microbenchmark win only counts if per-layer overhead does not eat it:

| | weights | decode step | tokens/s | under CUDA graph |
|---|---|---|---|---|
| fp16 (cuBLAS) | 13.0 GiB | 12.72 ms | 78.6 | 85.8 |
| **nibblegemm INT4** | **3.35 GiB** | **5.14 ms** | **194.7** | **227.5** |

**2.48x eager, 2.65x under CUDA graph, on 3.88x less weight memory.** The full
stack lands slightly under the best single-shape number because a decode step is
a *mix*: the narrow GQA k/v projections (4096×1024) and the 4096×4096 q/o
projections are the shapes that sit furthest from peak.

**Quality**, on a real checkpoint (TinyLlama-1.1B, perplexity over 4096 tokens
of held-out prose):

| | weights | perplexity | vs fp16 |
|---|---|---|---|
| fp16 | 2.05 GiB | 11.355 | — |
| INT4 (max-abs RTN) | 0.71 GiB | 12.760 | +12.4% |
| INT4 + MSE clip search | 0.71 GiB | 13.947 | +22.8% |

The kernel contributes **none** of that error — it is bit-exact against the
reference. All 12.4% is the quantisation *scheme*: round-to-nearest with no
calibration, which is the weakest algorithm in the family. GPTQ or AWQ weights
would slot into the same kernel unchanged and recover most of it.

That third row is a negative result worth keeping. Searching clipping ratios to
minimise weight MSE **cut weight RMSE by 12% and made the model 23% worse**,
because a squared-error criterion is happiest to clip exactly the
large-magnitude weights transformer outputs depend on. It is a clean
demonstration of why AWQ weights error by *activation* magnitude and GPTQ uses
second-order information: weight error is a proxy, and optimising the proxy
moved the real metric the wrong way. The search is still in the code, off by
default.

One caveat stated plainly: the 1.1B decode-throughput numbers in that run are
overhead-dominated — at 22 layers in eager HuggingFace, Python dispatch swamps
the kernel and the fp16/INT4 difference is inside the noise. The 8B table above
is the meaningful speed measurement; the 1.1B run is there for quality.

---

## The optimisation ladder

Five kernels, each changing exactly one thing, so every speedup is attributable.
Shape 4096×14336, M=1:

| stage | change | ms | GB/s | vs previous |
|---|---|---|---|---|
| v0 | naive: thread per output, scalar decode | 0.537 | 56 | — |
| v1 | reuse each packed word for all 8 nibbles | 0.299 | 102 | 1.80x |
| v2 | `lop3` bit-trick decode + half2 FMA | 0.073 | 416 | 4.09x |
| v3 | 128-bit loads, 4 columns per thread | 0.132 | 229 | **0.55x** |
| v4 | split-K across the reduction axis | 0.031 | 977 | 4.28x |

**17.3x end to end.** Two of these are worth reading about:

**v2 — dequantisation without a conversion instruction.** `0x6400` is exactly
`1024.0` in fp16, and at that exponent the mantissa bits are integer-valued. So
OR-ing a 4-bit nibble into `0x6400` yields exactly the float `1024 + n` with no
integer-to-float convert at all, and recovering `n` is a single packed subtract
handling two weights at once. One `lop3.b32` extracts two nibbles 16 bits apart
directly into `half2` layout. Eight weights cost 8 instructions instead of ~32,
and the quantiser's zero-point folds into a bias that was already being applied.
It is also **bit-exact** — the tests sweep all 16 values in all 8 nibble slots
and demand `torch.equal` against a scalar reference, because a wrong magic
constant here would silently degrade a model rather than fail loudly.

**v3 — the rung that made things slower.** Textbook vectorisation cost 45%. Four
output columns per thread meant 512 columns per block, so a 4096-column layer
produced 8 blocks on a 108-SM GPU. The kernel was never issue-limited; it was
starved of parallelism, and vectorising starved it further. That is precisely
why v4 (split-K) exists — worth 4.6x–13x on its own. The failed rung is kept in
the repo rather than quietly deleted.

The log also records a **second hypothesis that measurement killed**: narrowing
the block to raise occupancy does nothing, because total warps are
`(N / TN) × splits` regardless of how they are grouped into blocks.

---

## How the benchmarks avoid lying

Two decisions in [`bench/harness.py`](bench/harness.py) matter more than any
kernel change:

**L2 defeat.** A 4096×4096 INT4 weight matrix is 8 MB; the A100's L2 is 40 MB.
Timing the same buffer in a loop serves everything after the first iteration
from L2 at several TB/s. That does not merely inflate results — it inflates them
*asymmetrically*, since 8 MB of INT4 weights fit in L2 and the 32 MB fp16
baseline does not, manufacturing most of a speedup from nothing. The harness
rotates through enough distinct weight copies to overflow L2, which is also what
a real forward pass does: every layer is touched once, never revisited.

**Measured peak, not the datasheet.** All percentages are against a measured
1375 GB/s, not the unreachable 1555 GB/s spec figure.

Also: median with p5/p95 across trials, SM clocks sampled before and after so a
throttled run is visible rather than silently averaged in, and a CUDA-graph mode
that separates launch overhead from kernel time.

---

## Correctness

173 tests, all passing.

- **Bit-exactness** of the `lop3` decode against a scalar CUDA path and a NumPy
  reference, sweeping every nibble value in every slot.
- **Format pinning**: the nibble interleave is asserted against a hard-coded
  packed word, so a layout change cannot silently desynchronise the packer from
  the kernels.
- **Tolerances derived, not guessed** — each one is justified by emulating the
  kernel's fp16 accumulation window on CPU, with the margin stated in a comment.
- **Error attributed to quantisation, not the kernel**: `QuantLinear` tests
  assert the kernel contributes under 0.1% while INT4 quantisation itself
  contributes ~10%.
- Edge shapes (M and N not multiples of the block tile), every split-K width,
  and both group sizes.

```bash
python -m pytest tests/ -q     # 173 passed
```

---

## Two regimes, one dispatch

| | decode (M ≤ 8) | prefill (M > 8) |
|---|---|---|
| bottleneck | HBM bandwidth | memory latency, not issue rate |
| kernel | split-K streaming GEMV | 128×128 tile, `cp.async` + register-pipelined weights |
| metric | GB/s vs measured peak | TFLOP/s |
| result | **3.13x over cuBLAS fp16** | 59–64% of cuBLAS fp16 |

**Prefill is still the weaker half, and the repo says so.** Dequantising to fp16
and calling cuBLAS reaches 99–100% of fp16 throughput; the fused kernel reaches
59–64%. It earns its place by never materialising the fp16 weight matrix — it
saves the memory, which is why INT4 exists — and `ops.matmul` dispatches on
batch size rather than pretending otherwise.

It got there by testing five hypotheses, of which **two were wrong**
([full write-up](docs/OPTIMIZATION_LOG.md#prefill-regime-large-m-arithmetic-intensity-decides)):

| hypothesis | outcome | TFLOP/s |
|---|---|---|
| the tile is too small — arithmetic intensity caps it | confirmed | 73.8 → 100.1 |
| wmma's fragment addressing is the overhead | **disproved** | 100.1 |
| a runtime integer division sits on the hot path | confirmed | 100.1 → 111.7 |
| 64-bit address arithmetic still costs | 21% fewer instructions, **no speedup** | 111.7 |
| the weight path is synchronous | confirmed | 111.7 → **150.0** |

The tile result was predicted before writing code: intensity is
`2·BM·BN·BK / (BM·BK·2 + BN·BK/2)` and `BK` cancels, so a 64×64 tile's
51 FLOP/byte against the ~220 the A100 needs caps it near 71 TFLOP/s — and the
first version measured 73.8, i.e. it was already at its roofline.

The two failures are what located the real problem. Rewriting the inner loop in
raw `mma.sync` + `ldmatrix` produced a **bit-identical** kernel that was not
faster, because nvcc already lowers wmma to much the same sequence. Then an
`ncu` pipe breakdown found 64% of instructions on the ALU/FMA pipes (where
integer address arithmetic lives) plus a stray occupied XU pipe — an integer
division by a runtime `group_size`, which templating took to exactly zero for
+11%. But stripping a further 21% of instructions after that bought *nothing*,
which proved the kernel was not issue-bound at all. It was stalling.

On what, specifically: activations stream through `cp.async`, but weights must
be **decoded** before they can be stored, so the weight path was a plain global
load followed by a stall, per thread, per tile. Splitting staging into a
register prefetch and a commit placed a full tile apart gives every weight load
~128 `mma` instructions of cover. Occupancy did not change and the instruction
count barely did, but SM throughput went from 37% to 49% — the signature of
covered latency rather than less work — and throughput rose 34%.

**Against Triton, the kernel now leads in both regimes**: 1.34x at prefill
(150.0 vs 111.6 TFLOP/s) and 4.5x at decode. Before that last change prefill was
a dead heat, which was itself the useful signal — prefill is a well-trodden
tiled-GEMM problem where an autotuner sweeping block sizes is hard to beat by
hand, and it took a structural change to the pipeline rather than a tuning knob
to get ahead. Decode is the opposite: an awkward, launch-geometry-sensitive,
bandwidth-bound shape where `tl.dot`'s 16-row minimum tile wastes 15/16 of the
work at M=1.

---

## Quick start

Requires an Ampere GPU (sm_80), CUDA 12.x, PyTorch. The extension is JIT-built
on first use — there is nothing to install.

```python
import torch, nibblegemm as ng

W = torch.randn(4096, 14336)                  # [K, N], consumed as Y = X @ W
qw = ng.quantize(W, group_size=128).to("cuda")  # packed INT4 + fp16 scales
X  = torch.randn(1, 4096, dtype=torch.float16, device="cuda")

Y = ng.matmul(X, qw)                           # dispatches on batch size
```

Swap it into a model:

```python
qlinear = ng.QuantLinear.from_linear(model.mlp.gate_proj, group_size=128)
```

Reproduce the numbers:

```bash
python scripts/smoke.py              # build + correctness
python bench/bench_decode.py         # the ladder and its baselines
python bench/bench_prefill.py        # tensor-core path + crossover sweep
python bench/autotune.py             # split-K / block-width sweep
python bench/launch_overhead.py      # eager vs CUDA graph
python bench/bench_model.py layers   # decode step at Llama-3-8B scale
python bench/bench_model.py model    # perplexity on a real checkpoint
```

No local GPU? [`tools/colab_run.py`](tools/colab_run.py) ships the working tree
to a Colab A100 and runs a command there — this project was developed on an
Apple M4 Pro that way.

---

## Layout

```
csrc/dequant.cuh        the lop3 bit-trick decode, with the derivation
csrc/mma.cuh            mma.sync / ldmatrix wrappers + the fragment layouts
csrc/gemv_w4a16.cu      decode kernels v0-v4
csrc/gemm_w4a16.cu      prefill kernels v6 (wmma) and v7 (mma.sync)
python/nibblegemm/      quantiser (defines the on-device format), ops, Triton baseline
bench/                  harness, kernel benchmarks, end-to-end model benchmark
tests/                  173 tests
docs/OPTIMIZATION_LOG.md   every measurement, including the ones that disproved me
```

## Scope

Ampere (sm_80) only; symmetric INT4 with group size 64 or 128; `N` a multiple of
4; decode path up to M=8. Ada and Hopper would need FP8 and `wgmma` paths
respectively. Not a drop-in replacement for Marlin or Machete — those are
further along, and this is a from-scratch implementation built to understand and
measure the problem rather than to displace them.

## Licence

MIT.
