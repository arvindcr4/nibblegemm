"""Triton W4A16 matmul -- the baseline the CUDA kernels have to justify themselves against.

Triton is what most people reach for first when they need a weight-only INT4
matmul, so it is the honest comparison point. Beating ``torch.matmul`` on a
dequantised weight proves very little: that path re-materialises the whole fp16
weight and pays for the bandwidth twice. Beating a competently written Triton
kernel -- tiled, ``tl.dot``-based, autotuned -- is the actual claim ``csrc/``
makes, so this module exists to keep that claim falsifiable.

It reads exactly the format ``quant.py`` defines, not a convenient
reformulation, so both paths touch the same bytes:

* ``qweight`` is ``int32[K // 8, N]``; eight nibbles along ``K`` per word.
* Slot order is interleaved, ``SLOT_FOR_K = [0, 4, 1, 5, 2, 6, 3, 7]``: slot
  ``s < 4`` holds ``k = 2s``, slot ``s >= 4`` holds ``k = 2(s - 4) + 1``. The
  kernel undoes this by deriving the shift from ``k`` rather than assuming
  ascending nibbles.
* ``scales`` is ``fp16[K // G, N]``; dequantisation is ``(q_u - 8) * scale``
  with no zero-point.

Tiling constraint: ``BLOCK_K`` must be a multiple of 8 (a packed word is
indivisible) and must divide ``group_size`` (so one weight tile is covered by a
single row of scales, which is what makes the scale a cheap ``[1, BLOCK_N]``
broadcast instead of a per-element gather). Both are enforced by
``tl.static_assert`` at compile time.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from .quant import QuantizedWeight

try:  # Triton is optional: absence must surface at call time, not import time.
    import triton
    import triton.language as tl

    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - depends on the host
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc

__all__ = [
    "HAS_TRITON",
    "matmul",
    "triton_w4a16_matmul",
    "triton_w4a16_matmul_fixed",
]

HAS_TRITON: bool = _IMPORT_ERROR is None

#: Program-id swizzle width for L2 reuse; see the Triton matmul tutorial.
_GROUP_M = 8

PACK_FACTOR = 8
ZERO_POINT = 8


def _require_triton() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "the Triton W4A16 baseline needs `triton` installed on a CUDA machine"
        ) from _IMPORT_ERROR


def _prune_configs(configs: Sequence[Any], named_args: Dict[str, Any], *_: Any, **kwargs: Any):
    """Drop configs whose BLOCK_K does not divide the runtime group size."""
    g = named_args.get("GROUP_SIZE", kwargs.get("GROUP_SIZE"))
    if g is None:  # older Triton may not surface constexprs here; shipped configs are safe
        return list(configs)
    viable = [c for c in configs if g % c.kwargs["BLOCK_K"] == 0]
    if not viable:
        raise ValueError(
            f"no autotune config has BLOCK_K dividing group_size={g}; "
            "use triton_w4a16_matmul_fixed with a compatible block_k"
        )
    return viable


if _IMPORT_ERROR is None:

    _CONFIGS: List[Any] = [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ]

    @triton.jit
    def _w4a16_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        y_ptr,
        M,
        N,
        K,
        stride_xm,
        stride_xk,
        stride_qk,
        stride_qn,
        stride_sg,
        stride_sn,
        stride_ym,
        stride_yn,
        GROUP_SIZE: tl.constexpr,
        GROUP_M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # A packed word is 8 k-values wide, and one weight tile must sit inside a
        # single quantisation group so its scales are one [1, BLOCK_N] row.
        tl.static_assert(BLOCK_K % 8 == 0, "BLOCK_K must be a multiple of 8")
        tl.static_assert(GROUP_SIZE % BLOCK_K == 0, "BLOCK_K must divide group_size")

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_rows = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_in_group = pid % num_pid_in_group
        pid_m = first_pid_m + (pid_in_group % group_rows)
        pid_n = pid_in_group // group_rows

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        m_mask = offs_m < M
        n_mask = offs_n < N

        # Undo the packing interleave: logical offset k lives in slot
        # k // 2 for even k and k // 2 + 4 for odd k, i.e. SLOT_FOR_K.
        # Every k0 below is a multiple of BLOCK_K and BLOCK_K % 8 == 0, so the
        # slot of a global k depends only on its position inside the tile.
        k_in_word = offs_k % 8
        shift = ((k_in_word // 2) + 4 * (k_in_word % 2)) * 4

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        q_ptrs = q_ptr + (offs_k[:, None] // 8) * stride_qk + offs_n[None, :] * stride_qn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_mask = (k0 + offs_k) < K
            # Out-of-range k decodes to -8 * scale rather than 0, which is
            # harmless: the matching activation lane is masked to 0.
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            packed = tl.load(q_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0)
            # Masking with 0xF after the shift makes int32 sign extension irrelevant.
            nib = (packed >> shift[:, None]) & 0xF
            scale = tl.load(
                s_ptr + (k0 // GROUP_SIZE) * stride_sg + offs_n * stride_sn,
                mask=n_mask,
                other=0.0,
            )
            w = (nib - 8).to(tl.float32) * scale[None, :].to(tl.float32)
            acc += tl.dot(x, w.to(tl.float16))
            x_ptrs += BLOCK_K * stride_xk
            q_ptrs += (BLOCK_K // 8) * stride_qk

        y_ptrs = y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
        tl.store(y_ptrs, acc.to(tl.float16), mask=m_mask[:, None] & n_mask[None, :])

    _autotuned_kernel = triton.autotune(
        configs=_CONFIGS,
        key=["M", "N", "K", "GROUP_SIZE"],
        prune_configs_by={"early_config_prune": _prune_configs},
    )(_w4a16_kernel)

else:  # pragma: no cover - no Triton on this host
    _CONFIGS = []


def _validate(
    X: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, group_size: int
) -> tuple[int, int, int]:
    if X.dim() != 2:
        raise ValueError(f"expected a 2-D activation matrix, got {tuple(X.shape)}")
    if qweight.dim() != 2 or scales.dim() != 2:
        raise ValueError("qweight and scales must both be 2-D")
    if X.dtype != torch.float16:
        raise TypeError(f"activations must be fp16, got {X.dtype}")
    if qweight.dtype != torch.int32:
        raise TypeError(f"qweight must be int32, got {qweight.dtype}")
    if scales.dtype != torch.float16:
        raise TypeError(f"scales must be fp16, got {scales.dtype}")
    if not (X.is_cuda and qweight.is_cuda and scales.is_cuda):
        raise ValueError("all operands must live on the same CUDA device")
    if group_size % PACK_FACTOR:
        raise ValueError(f"group_size={group_size} must be divisible by {PACK_FACTOR}")

    M = X.size(0)
    K = qweight.size(0) * PACK_FACTOR
    N = qweight.size(1)
    if X.size(1) != K:
        raise ValueError(f"K mismatch: activations have {X.size(1)}, weights have {K}")
    if K % group_size:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")
    if tuple(scales.shape) != (K // group_size, N):
        raise ValueError(
            f"scales must be [{K // group_size}, {N}], got {tuple(scales.shape)}"
        )
    return M, N, K


def triton_w4a16_matmul(
    X: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """``X @ dequant(qweight, scales)`` -> fp16 ``[M, N]``, autotuned."""
    _require_triton()
    M, N, K = _validate(X, qweight, scales, group_size)
    if not any(group_size % cfg.kwargs["BLOCK_K"] == 0 for cfg in _CONFIGS):
        raise ValueError(
            f"no autotune config has BLOCK_K dividing group_size={group_size}; "
            "use triton_w4a16_matmul_fixed with a compatible block_k"
        )

    y = torch.empty((M, N), device=X.device, dtype=torch.float16)
    if M == 0 or N == 0:
        return y

    def grid(meta: Dict[str, int]):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    _autotuned_kernel[grid](
        X,
        qweight,
        scales,
        y,
        M,
        N,
        K,
        X.stride(0),
        X.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        y.stride(0),
        y.stride(1),
        group_size,
        _GROUP_M,
    )
    return y


def triton_w4a16_matmul_fixed(
    X: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
) -> torch.Tensor:
    """Same kernel with a hand-picked tile: no autotuning, no config cache."""
    _require_triton()
    M, N, K = _validate(X, qweight, scales, group_size)
    if block_k % PACK_FACTOR:
        raise ValueError(f"block_k={block_k} must be a multiple of {PACK_FACTOR}")
    if group_size % block_k:
        raise ValueError(f"block_k={block_k} must divide group_size={group_size}")
    if min(block_m, block_n, block_k) < 16:
        raise ValueError("tl.dot requires every block dimension to be at least 16")

    y = torch.empty((M, N), device=X.device, dtype=torch.float16)
    if M == 0 or N == 0:
        return y

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _w4a16_kernel[grid](
        X,
        qweight,
        scales,
        y,
        M,
        N,
        K,
        X.stride(0),
        X.stride(1),
        qweight.stride(0),
        qweight.stride(1),
        scales.stride(0),
        scales.stride(1),
        y.stride(0),
        y.stride(1),
        group_size,
        _GROUP_M,
        block_m,
        block_n,
        block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y


def matmul(X: torch.Tensor, qw: QuantizedWeight) -> torch.Tensor:
    """Triton counterpart of :func:`nibblegemm.ops.matmul`."""
    return triton_w4a16_matmul(X, qw.qweight, qw.scales, qw.group_size)
