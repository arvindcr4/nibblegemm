"""Minimal driver that runs one kernel a handful of times, for ncu to attach to.

Kept deliberately thin: anything else on the timeline shows up as extra kernels
in the profile and has to be filtered back out.
"""
from __future__ import annotations

import argparse

import torch

import nibblegemm as ng


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["gemv", "gemm"], default="gemv")
    ap.add_argument("-M", type=int, default=1)
    ap.add_argument("-K", type=int, default=4096)
    ap.add_argument("-N", type=int, default=4096)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--gemm-version", type=int, default=6)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    W = torch.randn(args.K, args.N, dtype=torch.float32) * 0.02
    qw = ng.quantize(W, group_size=args.group_size).to("cuda")
    X = torch.randn(args.M, args.K, dtype=torch.float16, device="cuda")

    fn = (lambda: ng.gemv(X, qw, version=4)) if args.target == "gemv" else (lambda: ng.gemm(X, qw, version=args.gemm_version))
    fn()  # trigger the JIT build outside the profiled region
    torch.cuda.synchronize()

    for _ in range(args.iters):
        fn()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
