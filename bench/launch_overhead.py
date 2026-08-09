"""How much of the decode kernel's wall-clock is launch overhead, not memory?

The split-K decode kernel issues two launches per call (the main pass and the
partial reduction) and, on a small layer, finishes in well under 20 us. At that
scale launch cost is no longer a rounding error, and it changes what "slow"
means: a kernel at 40% of peak bandwidth because it is waiting on launches
needs a different fix than one at 40% because its access pattern is poor.

Capturing the same calls into a CUDA graph and replaying removes per-launch CPU
work while leaving the kernels untouched, so the eager/graph gap is a direct
measurement of what launching costs. This is also how the kernel would be driven
in a real serving loop, where the whole decode step is captured once.
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
COLUMNS = ["shape", "M", "eager ms", "graph ms", "launch us", "eager GB/s", "graph GB/s",
           "graph %peak"]


def run(K, N, M, group_size, peak):
    W = torch.randn(K, N, dtype=torch.float32) * 0.02
    qw = ng.quantize(W, group_size=group_size).to("cuda")
    rot = H.Rotation(qw.qweight)
    variants = H.Cycle(
        ng.QuantizedWeight(rot[i], qw.scales, group_size, K, N) for i in range(rot.count)
    )
    X = torch.randn(M, K, dtype=torch.float16, device="cuda")

    call = lambda i: ng.matmul(X, variants[i])  # noqa: E731
    eager = H.bench(call)
    graph = H.bench_graph(call)

    eager_gbps = H.achieved_gbps(M, K, N, group_size, 4, eager.median_ms)
    graph_gbps = H.achieved_gbps(M, K, N, group_size, 4, graph.median_ms)
    row = {
        "shape": f"{K}x{N}", "M": M,
        "eager ms": round(eager.median_ms, 4),
        "graph ms": round(graph.median_ms, 4),
        "launch us": round((eager.median_ms - graph.median_ms) * 1000, 2),
        "eager GB/s": round(eager_gbps, 1),
        "graph GB/s": round(graph_gbps, 1),
        "graph %peak": round(100 * graph_gbps / peak, 1),
    }
    del variants, rot, qw
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--batch", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default="docs/results/launch_overhead.csv")
    args = ap.parse_args()

    print(H.device_summary())
    ng.extension()
    peak = H.measured_peak_read_gbps()
    print(f"measured achievable bandwidth: {peak:.0f} GB/s\n")

    rows = [run(K, N, M, args.group_size, peak) for K, N in SHAPES for M in args.batch]
    print(H.markdown_table(rows, COLUMNS))
    H.write_csv(rows, args.out, COLUMNS)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
