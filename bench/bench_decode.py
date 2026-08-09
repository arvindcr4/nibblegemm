"""Decode-regime benchmark: the optimisation ladder and the baselines it must beat.

At small batch the kernel is a bandwidth problem, so the headline number is not
TFLOP/s but achieved GB/s against measured peak. The interesting ceiling is not
"as fast as possible" either -- it is the ratio of fp16 weight bytes to INT4
weight bytes. That ratio (~3.9x at group size 128) is the *most* a perfect
INT4 kernel can beat cuBLAS fp16 by in this regime, and reporting achieved
speedup against it says far more than a bare multiplier.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

import nibblegemm as ng  # noqa: E402

# (K, N) pairs drawn from real decoder layers: Llama-3-8B attention and MLP
# projections, plus a 70B-shaped square case.
SHAPES = [
    (4096, 4096),    # 8B  q/o proj
    (4096, 14336),   # 8B  gate/up proj
    (14336, 4096),   # 8B  down proj
    (8192, 8192),    # 70B attention proj
]
LADDER = [
    (0, "v0 naive"),
    (1, "v1 coalesced"),
    (2, "v2 fast-dequant"),
    (3, "v3 vec4+regblock"),
    (4, "v4 split-K"),
]
COLUMNS = ["shape", "M", "impl", "ms", "spread%", "GB/s", "%peak", "vs fp16", "%ceiling"]


def make_rotations(K, N, group_size, device="cuda"):
    """Quantised and fp16 copies of one weight, replicated to overflow L2."""
    W = (torch.randn(K, N, dtype=torch.float32) * 0.02)
    qw = ng.quantize(W, group_size=group_size).to(device)
    w16 = W.to(torch.float16).to(device)

    qrot = H.Rotation(qw.qweight)
    wrot = H.Rotation(w16)
    variants = H.Cycle(
        ng.QuantizedWeight(qrot[i], qw.scales, group_size, K, N) for i in range(qrot.count)
    )
    return variants, wrot, qrot


def run_shape(K, N, M_list, group_size, peak_gbps, versions, use_triton):
    variants, wrot, qrot = make_rotations(K, N, group_size)
    rows = []

    fp16_bytes = H.total_bytes(1, K, N, group_size, 16)
    int4_bytes = H.total_bytes(1, K, N, group_size, 4)
    ceiling = fp16_bytes / int4_bytes

    for M in M_list:
        X = torch.randn(M, K, dtype=torch.float16, device="cuda")

        def record(name, t, bits):
            gbps = H.achieved_gbps(M, K, N, group_size, bits, t.median_ms)
            rows.append({
                "shape": f"{K}x{N}", "M": M, "impl": name,
                "ms": round(t.median_ms, 4),
                "spread%": round(t.spread_pct, 1),
                "GB/s": round(gbps, 1),
                "%peak": round(100 * gbps / peak_gbps, 1),
                "_ms": t.median_ms, "_throttled": t.throttled,
            })

        # Baseline 1: dense fp16 matmul. This is the operation being replaced.
        t_fp16 = H.bench(lambda i: torch.mm(X, wrot[i]))
        record("torch fp16 (cuBLAS)", t_fp16, 16)
        base_ms = t_fp16.median_ms

        # Baseline 2: expand to fp16, then cuBLAS. The obvious way to "support"
        # INT4 weights, and the reason fusing the dequantisation matters -- it
        # pays the full fp16 bandwidth cost plus a materialisation pass.
        # Fewer reps: every call allocates and writes a whole fp16 matrix.
        t_dq = H.bench(lambda i: torch.mm(X, ng.dequant(variants[i])), reps=8, trials=15)
        record("dequant + cuBLAS", t_dq, 4)

        if use_triton:
            try:
                from nibblegemm import triton_ref
                t_tr = H.bench(lambda i: triton_ref.matmul(X, variants[i]), reps=16, trials=20)
                record("triton w4a16", t_tr, 4)
            except Exception as exc:  # pragma: no cover - optional dependency
                print(f"  [triton skipped: {type(exc).__name__}: {exc}]", file=sys.stderr)

        for v, label in LADDER:
            if v not in versions:
                continue
            # v0 is quadratically slower; keep its rep count low so a full sweep
            # does not spend minutes inside the strawman.
            reps, trials = (4, 10) if v == 0 else (32, 30)
            t = H.bench(lambda i, v=v: ng.gemv(X, variants[i], version=v), reps=reps, trials=trials)
            record(f"nibblegemm {label}", t, 4)

        for r in rows:
            if r["M"] == M and r["shape"] == f"{K}x{N}":
                r["vs fp16"] = f"{base_ms / r['_ms']:.2f}x"
                r["%ceiling"] = (round(100 * (base_ms / r['_ms']) / ceiling, 1)
                                 if r["impl"] != "torch fp16 (cuBLAS)" else "")

    del variants, wrot, qrot
    torch.cuda.empty_cache()
    return rows, ceiling


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--versions", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--quick", action="store_true", help="one shape, batch 1, final kernel only")
    ap.add_argument("--no-triton", action="store_true")
    ap.add_argument("--out", default="docs/results/decode.csv")
    args = ap.parse_args()

    shapes, batches, versions = SHAPES, args.batch, set(args.versions)
    if args.quick:
        shapes, batches, versions = SHAPES[:1], [1], {4}

    print(H.device_summary())
    ng.extension()
    peak = H.measured_peak_read_gbps()
    print(f"measured achievable bandwidth: {peak:.0f} GB/s "
          f"(copy benchmark; spec sheet says 1555 GB/s)\n")

    all_rows = []
    for K, N in shapes:
        rows, ceiling = run_shape(K, N, batches, args.group_size, peak, versions,
                                  not args.no_triton)
        print(f"--- {K}x{N}  (bandwidth ceiling vs fp16: {ceiling:.2f}x) ---")
        print(H.markdown_table(rows, COLUMNS))
        print()
        all_rows += rows

    throttled = [r for r in all_rows if r.get("_throttled")]
    if throttled:
        print(f"WARNING: SM clock moved during {len(throttled)} measurement(s); "
              "treat those rows as indicative only.")

    H.write_csv(all_rows, args.out, COLUMNS)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
