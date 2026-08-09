"""Prefill-regime benchmark, plus the decode/prefill crossover sweep.

Once M is large the weights are reused across rows, HBM stops being the limit,
and the comparison flips: cuBLAS fp16 is no longer handicapped by weight traffic
and is running highly tuned tensor-core code, so beating it on time is not the
goal and claiming otherwise would be dishonest. What INT4 buys at prefill is
*capacity* -- a quarter of the weight memory, which is what lets the model fit
at all -- and the kernel's job is to charge as little throughput as possible for
that.

So the headline here is achieved TFLOP/s against cuBLAS fp16 as a fraction, and
the interesting number is how close the fused kernel gets while reading a
quarter of the bytes.

The crossover sweep exists because the dispatch threshold in ops.matmul is a
claim about where the bottleneck changes, and a claim like that should be
measured rather than asserted.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

import nibblegemm as ng  # noqa: E402

# A100 SXM4 dense fp16 tensor-core peak, for context on the TFLOP/s column.
A100_FP16_TFLOPS = 312.0

SHAPES = [(4096, 4096), (4096, 14336)]
PREFILL_M = [128, 512, 2048, 4096]
CROSSOVER_M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

PREFILL_COLUMNS = ["shape", "M", "impl", "ms", "TFLOP/s", "%fp16", "%A100 peak"]
CROSSOVER_COLUMNS = ["shape", "M", "decode ms", "prefill ms", "faster", "fp16 ms", "vs fp16"]


def build(K, N, group_size):
    W = torch.randn(K, N, dtype=torch.float32) * 0.02
    qw = ng.quantize(W, group_size=group_size).to("cuda")
    rot = H.Rotation(qw.qweight)
    variants = H.Cycle(
        ng.QuantizedWeight(rot[i], qw.scales, group_size, K, N) for i in range(rot.count)
    )
    wrot = H.Rotation(W.to(torch.float16).cuda())
    return qw, variants, wrot


def prefill(K, N, group_size, use_triton):
    qw, variants, wrot = build(K, N, group_size)
    rows = []
    for M in PREFILL_M:
        X = torch.randn(M, K, dtype=torch.float16, device="cuda")

        # Correctness at every measured point; a fast wrong kernel is worthless.
        ref = ng.reference_matmul(X, variants[0])
        for v in (6, 7):
            y = ng.gemm(X, variants[0], version=v)
            err = (y.float() - ref.float()).abs().max().item() / ref.float().abs().mean().item()
            assert err < 5e-2, f"gemm v{v} wrong at M={M} (rel err {err:.2e})"

        def add(name, ms):
            tf = H.tflops(M, K, N, ms)
            rows.append({"shape": f"{K}x{N}", "M": M, "impl": name,
                         "ms": round(ms, 4), "TFLOP/s": round(tf, 1),
                         "%A100 peak": round(100 * tf / A100_FP16_TFLOPS, 1), "_ms": ms})

        base = H.bench(lambda i: torch.mm(X, wrot[i]), reps=8, trials=20).median_ms
        add("torch fp16 (cuBLAS)", base)
        add("nibblegemm v6 (wmma)",
            H.bench(lambda i: ng.gemm(X, variants[i], version=6), reps=8, trials=20).median_ms)
        add("nibblegemm v7 (mma.sync)",
            H.bench(lambda i: ng.gemm(X, variants[i], version=7), reps=8, trials=20).median_ms)
        add("dequant + cuBLAS",
            H.bench(lambda i: torch.mm(X, ng.dequant(variants[i])), reps=4, trials=12).median_ms)
        if use_triton:
            try:
                from nibblegemm import triton_ref
                add("triton w4a16",
                    H.bench(lambda i: triton_ref.matmul(X, variants[i]), reps=8,
                            trials=20).median_ms)
            except Exception as exc:
                print(f"  [triton skipped: {type(exc).__name__}: {exc}]", file=sys.stderr)

        for r in rows:
            if r["M"] == M and r["shape"] == f"{K}x{N}":
                r["%fp16"] = f"{100 * base / r['_ms']:.0f}%"

    del variants, wrot, qw
    torch.cuda.empty_cache()
    return rows


def crossover(K, N, group_size):
    qw, variants, wrot = build(K, N, group_size)
    rows = []
    for M in CROSSOVER_M:
        X = torch.randn(M, K, dtype=torch.float16, device="cuda")
        dec = (H.bench(lambda i: ng.gemv(X, variants[i], version=4), reps=16, trials=20).median_ms
               if M <= ng.DECODE_MAX_M else None)
        pre = H.bench(lambda i: ng.gemm(X, variants[i]), reps=16, trials=20).median_ms
        fp16 = H.bench(lambda i: torch.mm(X, wrot[i]), reps=16, trials=20).median_ms
        best = min(v for v in (dec, pre) if v is not None)
        rows.append({
            "shape": f"{K}x{N}", "M": M,
            "decode ms": round(dec, 4) if dec else "n/a",
            "prefill ms": round(pre, 4),
            "faster": ("decode" if dec and dec < pre else "prefill"),
            "fp16 ms": round(fp16, 4),
            "vs fp16": f"{fp16 / best:.2f}x",
        })
    del variants, wrot, qw
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--skip-crossover", action="store_true")
    args = ap.parse_args()

    print(H.device_summary())
    ng.extension()
    print()

    all_rows = []
    for K, N in SHAPES:
        rows = prefill(K, N, args.group_size, not args.no_triton)
        print(f"--- prefill {K}x{N} ---")
        print(H.markdown_table(rows, PREFILL_COLUMNS))
        print()
        all_rows += rows
    H.write_csv(all_rows, "docs/results/prefill.csv", PREFILL_COLUMNS)

    if not args.skip_crossover:
        cross = []
        for K, N in SHAPES:
            rows = crossover(K, N, args.group_size)
            print(f"--- decode/prefill crossover {K}x{N} ---")
            print(H.markdown_table(rows, CROSSOVER_COLUMNS))
            print()
            cross += rows
        H.write_csv(cross, "docs/results/crossover.csv", CROSSOVER_COLUMNS)

    print("wrote docs/results/prefill.csv and crossover.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
