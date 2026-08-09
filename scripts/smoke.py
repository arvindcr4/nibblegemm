"""Fast build + correctness check. Run this first on any new machine."""
from __future__ import annotations

import sys

import numpy as np
import torch

import nibblegemm as ng


def section(msg):
    print(f"\n=== {msg} ===", flush=True)


def main() -> int:
    section("device")
    assert torch.cuda.is_available(), "no CUDA device"
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  sm_{p.major}{p.minor}  {p.multi_processor_count} SMs")

    section("pack / unpack roundtrip (cpu)")
    rng = np.random.default_rng(0)
    qu = rng.integers(0, 16, size=(256, 64), dtype=np.uint8)
    back = ng.unpack_nibbles(ng.pack_nibbles(qu), 256)
    assert np.array_equal(qu, back), "nibble pack/unpack is not an involution"
    print("ok")

    section("build extension")
    ng.extension()
    print("ok")

    section("dequant: fast lop3 path vs scalar path")
    K, N, G = 512, 128, 128
    W = torch.randn(K, N, dtype=torch.float32)
    qw = ng.quantize(W, group_size=G).to("cuda")
    fast = ng.dequant(qw, fast=True)
    slow = ng.dequant(qw, fast=False)
    ref = ng.dequantize(qw).cuda()
    print(f"fast vs scalar : max|d| = {(fast - slow).abs().max().item():.3e}")
    print(f"fast vs numpy  : max|d| = {(fast - ref).abs().max().item():.3e}")
    assert torch.equal(fast, slow), "lop3 decode disagrees with the scalar decode"
    assert torch.equal(fast, ref), "lop3 decode disagrees with the numpy reference"
    print("bit-exact")

    section("gemv ladder vs fp32 reference")
    K, N = 4096, 4096
    W = torch.randn(K, N, dtype=torch.float32) * 0.02
    qw = ng.quantize(W, group_size=128).to("cuda")
    for M in (1, 2, 4, 8):
        X = torch.randn(M, K, dtype=torch.float16, device="cuda")
        ref = ng.reference_matmul(X, qw)
        scale = ref.float().abs().mean().item()
        for v in (0, 1, 2, 3, 4):
            y = ng.gemv(X, qw, version=v)
            err = (y.float() - ref.float()).abs().max().item() / scale
            flag = "ok" if err < 5e-2 else "FAIL"
            print(f"  M={M} v{v}: rel max err {err:.2e}  {flag}")
            if err >= 5e-2:
                return 1

    section("gemm (tensor core) vs fp32 reference")
    for M in (64, 128, 512, 1000):
        X = torch.randn(M, K, dtype=torch.float16, device="cuda")
        ref = ng.reference_matmul(X, qw)
        y = ng.gemm(X, qw)
        err = (y.float() - ref.float()).abs().max().item() / ref.float().abs().mean().item()
        flag = "ok" if err < 5e-2 else "FAIL"
        print(f"  M={M}: rel max err {err:.2e}  {flag}")
        if err >= 5e-2:
            return 1

    print("\nall smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
