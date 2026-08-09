"""Python surface for the CUDA kernels.

The extension is JIT-built on first use via ``torch.utils.cpp_extension.load``,
which keeps the repo dependency-free: clone it onto any Ampere box with a CUDA
toolchain and the first call compiles.

Note that ``--use_fast_math`` is deliberately *not* passed. It would let the
compiler contract and reassociate float ops, which would make the accuracy
numbers in tests/test_gemv.py describe a different kernel than the one being
benchmarked. The dequantisation path is already exact; there is nothing to gain.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import torch

from .quant import QuantizedWeight, quantize  # noqa: F401  (re-exported)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CSRC = os.path.join(_ROOT, "csrc")
_SOURCES = ["bindings.cpp", "gemv_w4a16.cu", "gemm_w4a16.cu", "dequant_kernel.cu"]

_ext = None
_lock = threading.Lock()

#: Batch sizes at or below this go to the bandwidth-bound decode kernel.
DECODE_MAX_M = 8


def extension():
    """Build (once) and return the compiled CUDA extension."""
    global _ext
    if _ext is not None:
        return _ext
    with _lock:
        if _ext is None:
            from torch.utils.cpp_extension import load

            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
            _ext = load(
                name="nibblegemm_cuda",
                sources=[os.path.join(_CSRC, s) for s in _SOURCES],
                extra_cflags=["-O3", "-std=c++17"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++17",
                    "--expt-relaxed-constexpr",
                    "-lineinfo",  # keeps ncu source attribution usable
                ],
                extra_include_paths=[_CSRC],
                verbose=os.environ.get("NIBBLE_VERBOSE", "0") == "1",
            )
    return _ext


def dequant(qw: QuantizedWeight, fast: bool = True) -> torch.Tensor:
    """Expand packed INT4 back to a dense ``[K, N]`` fp16 matrix on device."""
    return extension().dequant_w4(qw.qweight, qw.scales, qw.group_size, fast)


def gemv(X: torch.Tensor, qw: QuantizedWeight, version: int = 4, splits: int = 0,
         block_threads: int = 0) -> torch.Tensor:
    """Decode-regime matmul. ``version`` selects a rung of the optimisation ladder.

    ``splits`` and ``block_threads`` are launch-geometry overrides; 0 means "let
    the built-in heuristic decide". They exist so bench/autotune.py can measure
    how much the heuristic leaves on the table.
    """
    return extension().gemv_w4a16(
        X, qw.qweight, qw.scales, qw.group_size, version, splits, block_threads
    )


def gemm(X: torch.Tensor, qw: QuantizedWeight, version: int = 6) -> torch.Tensor:
    """Prefill-regime tensor-core matmul.

    ``version`` selects the inner-product implementation: 6 is the wmma
    version, 7 is raw ``mma.sync`` + ``ldmatrix``. Both use the same 128x128
    tile and produce bit-identical results, so the gap between them is purely
    the cost of the wmma abstraction -- which measurement puts at roughly zero.
    6 is the default because it is marginally the faster of the two.
    """
    return extension().gemm_w4a16(X, qw.qweight, qw.scales, qw.group_size, version)


def matmul(X: torch.Tensor, qw: QuantizedWeight, splits: int = 0,
           block_threads: int = 0) -> torch.Tensor:
    """``X @ dequant(qw)``, dispatching on batch size.

    The crossover is not a tuning knob so much as a statement about which
    resource is saturated: below it the kernel is starved for bytes, above it for
    tensor-core issue slots. bench/bench_prefill.py measures where the two
    curves actually meet on this machine.
    """
    if X.dim() != 2:
        raise ValueError(f"expected a 2-D activation matrix, got {tuple(X.shape)}")
    if X.size(1) != qw.K:
        raise ValueError(f"K mismatch: activations have {X.size(1)}, weights have {qw.K}")
    if X.size(0) <= DECODE_MAX_M:
        return gemv(X, qw, version=4, splits=splits, block_threads=block_threads)
    return gemm(X, qw)


class QuantLinear(torch.nn.Module):
    """Drop-in replacement for ``nn.Linear`` with INT4 weights.

    Exists so the kernels can be dropped into a real model and measured
    end-to-end rather than only in a microbenchmark.
    """

    def __init__(self, qw: QuantizedWeight, bias: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("qweight", qw.qweight)
        self.register_buffer("scales", qw.scales)
        self.register_buffer("bias", bias if bias is not None else None)
        self.group_size, self.K, self.N = qw.group_size, qw.K, qw.N

    @classmethod
    def from_linear(cls, linear: torch.nn.Linear, group_size: int = 128) -> "QuantLinear":
        # nn.Linear stores [out, in]; the kernels consume [in, out].
        qw = quantize(linear.weight.data.t().contiguous(), group_size)
        bias = linear.bias.data.clone() if linear.bias is not None else None
        return cls(qw.to(linear.weight.device), bias)

    def _qw(self) -> QuantizedWeight:
        return QuantizedWeight(self.qweight, self.scales, self.group_size, self.K, self.N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        y = matmul(x.reshape(-1, shape[-1]).contiguous(), self._qw())
        y = y.reshape(*shape[:-1], self.N)
        return y + self.bias if self.bias is not None else y
