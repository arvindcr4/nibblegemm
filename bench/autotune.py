"""Sweep the decode kernel's launch geometry and report what the heuristic misses.

Two knobs interact. Split-K divides the reduction axis, but only K/group_size
ways -- so on a narrow layer that ceiling is hit while the grid is still too
small to saturate HBM. The only remaining source of parallelism is a narrower
block, which trades per-thread work for more blocks. Sweeping either knob alone
gives a misleading picture of the other, so this sweeps the product.

Output doubles as the evidence for the numbers quoted in docs/OPTIMIZATION_LOG.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

import nibblegemm as ng  # noqa: E402

SHAPES = [(4096, 4096), (4096, 14336), (14336, 4096), (8192, 8192)]
COLUMNS = ["shape", "M", "threads", "splits", "ms", "GB/s", "%peak", "vs heuristic"]


def sweep(K, N, M, group_size, peak, split_candidates, thread_candidates):
    W = torch.randn(K, N, dtype=torch.float32) * 0.02
    qw = ng.quantize(W, group_size=group_size).to("cuda")
    rot = H.Rotation(qw.qweight)
    variants = H.Cycle(
        ng.QuantizedWeight(rot[i], qw.scales, group_size, K, N) for i in range(rot.count)
    )
    X = torch.randn(M, K, dtype=torch.float16, device="cuda")

    ref = ng.reference_matmul(X, variants[0])
    rows = []

    def measure(splits, threads):
        t = H.bench(lambda i: ng.gemv(X, variants[i], version=4, splits=splits,
                                      block_threads=threads))
        return t.median_ms

    auto_ms = measure(0, 0)  # 0/0 = let the built-in heuristics decide
    max_useful = K // group_size

    def record(splits, threads, ms):
        gbps = H.achieved_gbps(M, K, N, group_size, 4, ms)
        rows.append({
            "shape": f"{K}x{N}", "M": M,
            "threads": "auto" if threads == 0 else threads,
            "splits": "auto" if splits == 0 else splits,
            "ms": round(ms, 4),
            "GB/s": round(gbps, 1),
            "%peak": round(100 * gbps / peak, 1),
            "vs heuristic": f"{auto_ms / ms:.2f}x",
            "_ms": ms,
        })

    record(0, 0, auto_ms)
    for threads in thread_candidates:
        for s in [c for c in split_candidates if c <= max_useful]:
            # Correctness is re-checked at every geometry: the split reduction is
            # the part most likely to break, and a fast wrong answer is worthless.
            y = ng.gemv(X, variants[0], version=4, splits=s, block_threads=threads)
            err = (y.float() - ref.float()).abs().max().item() / ref.float().abs().mean().item()
            assert err < 5e-2, f"threads={threads} splits={s} wrong (rel err {err:.2e})"
            record(s, threads, measure(s, threads))

    del variants, rot, qw
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--splits", type=int, nargs="+", default=[8, 16, 32, 48, 64, 112])
    ap.add_argument("--threads", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--top", type=int, default=6, help="rows to print per shape")
    ap.add_argument("--out", default="docs/results/autotune.csv")
    args = ap.parse_args()

    print(H.device_summary())
    ng.extension()
    peak = H.measured_peak_read_gbps()
    print(f"measured achievable bandwidth: {peak:.0f} GB/s\n")

    all_rows = []
    for K, N in SHAPES:
        for M in args.batch:
            rows = sweep(K, N, M, args.group_size, peak, args.splits, args.threads)
            best = min(rows, key=lambda r: r["_ms"])
            auto = rows[0]
            ranked = sorted(rows, key=lambda r: r["_ms"])[: args.top]
            print(f"--- {K}x{N} M={M}  (top {args.top} of {len(rows)} geometries) ---")
            print(H.markdown_table(ranked, COLUMNS))
            print(f"  heuristic: {auto['ms']} ms | best: threads={best['threads']} "
                  f"splits={best['splits']} at {best['ms']} ms "
                  f"({100 * (auto['_ms'] / best['_ms'] - 1):.1f}% left on the table)\n")
            all_rows += rows

    H.write_csv(all_rows, args.out, COLUMNS)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
