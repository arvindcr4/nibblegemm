"""Reference group-wise INT4 weight quantiser, packer, and dequantiser.

This module defines the on-disk/on-device data format that every kernel in
`csrc/` agrees with. It is deliberately written in plain NumPy/PyTorch and
optimised for being *obviously correct* rather than fast -- the CUDA kernels are
tested against it.

Format
------
Weights ``W`` have shape ``[K, N]`` and are consumed as ``Y = X @ W``, so ``K``
is the reduction dimension.

* **Grouping.** Quantisation groups run along ``K`` with size ``G`` (default
  128). Group ``g`` of column ``n`` gets one fp16 scale.
* **Numeric scheme.** Symmetric INT4: ``scale = max|w| / 7`` and
  ``q = clamp(round(w / scale), -8, 7)``. Stored biased into ``[0, 15]`` as
  ``q + 8``; dequantisation is ``(q_u - 8) * scale``. There is no per-group
  zero-point, which matches how GPTQ/AWQ symmetric checkpoints are shipped and
  lets the kernel fold the ``-8`` into a constant.
* **Packing.** Eight INT4 values along ``K`` share one ``uint32``, giving
  ``qweight[K // 8, N]``. Consecutive ``n`` stay adjacent in memory, so a warp
  walking ``n`` issues perfectly coalesced 32-bit (or vectorised 128-bit) loads.

Nibble interleave
-----------------
The eight nibbles are **not** stored in ``k`` order. The kernel decodes them
with four ``lop3.b32`` ops that each pull two nibbles sitting 16 bits apart
(masks ``0x000f000f`` / ``0x00f000f0``, applied to ``q`` and ``q >> 8``). Those
ops emit slots in the order ``0,4,1,5,2,6,3,7``. Storing ``k`` in that same
permuted order means the decoded registers land in ascending ``k`` with zero
shuffling -- the permutation cost is paid once, offline, instead of per load.

    slot s < 4  holds  k = 2s          (even k)
    slot s >= 4 holds  k = 2(s - 4)+1  (odd  k)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

#: ``SLOT_FOR_K[k]`` is the nibble slot that stores logical offset ``k`` (0..7).
SLOT_FOR_K = np.array([0, 4, 1, 5, 2, 6, 3, 7], dtype=np.int64)
#: Inverse permutation: ``K_FOR_SLOT[s]`` is the logical offset held by slot ``s``.
K_FOR_SLOT = np.argsort(SLOT_FOR_K)

QMIN, QMAX = -8, 7
ZERO_POINT = 8
PACK_FACTOR = 8


@dataclass(frozen=True)
class QuantizedWeight:
    """Packed INT4 weights plus their fp16 group scales."""

    qweight: torch.Tensor  # int32 [K // 8, N], nibble-interleaved
    scales: torch.Tensor  # fp16  [K // G, N]
    group_size: int
    K: int
    N: int

    def to(self, device) -> "QuantizedWeight":
        return QuantizedWeight(
            self.qweight.to(device), self.scales.to(device), self.group_size, self.K, self.N
        )

    @property
    def weight_bytes(self) -> int:
        """Bytes of weight traffic a perfect kernel must read (packed + scales)."""
        return self.qweight.numel() * 4 + self.scales.numel() * 2


def quantize(W: torch.Tensor, group_size: int = 128) -> QuantizedWeight:
    """Quantise a ``[K, N]`` weight matrix to packed group-wise INT4."""
    if W.ndim != 2:
        raise ValueError(f"expected a 2-D weight, got shape {tuple(W.shape)}")
    K, N = W.shape
    if K % group_size:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")
    if group_size % PACK_FACTOR:
        raise ValueError(f"group_size={group_size} must be divisible by {PACK_FACTOR}")

    w = W.detach().float().cpu()
    groups = w.reshape(K // group_size, group_size, N)

    # Symmetric scale. Clamp away exact zeros so all-zero groups stay finite.
    scale = (groups.abs().amax(dim=1) / QMAX).clamp(min=1e-8)  # [K/G, N]
    q = torch.round(groups / scale.unsqueeze(1)).clamp(QMIN, QMAX)
    qu = (q + ZERO_POINT).reshape(K, N).to(torch.uint8).numpy()  # values in [0, 15]

    return QuantizedWeight(
        qweight=torch.from_numpy(pack_nibbles(qu)),
        scales=scale.to(torch.float16).contiguous(),
        group_size=group_size,
        K=K,
        N=N,
    )


def pack_nibbles(qu: np.ndarray) -> np.ndarray:
    """Pack ``[K, N]`` values in ``[0, 15]`` into ``int32[K // 8, N]``.

    Uses the interleaved slot order documented at module level. Packing is done
    in ``uint32`` so the top nibble (bits 28..31) cannot trip signed overflow,
    then reinterpreted as ``int32`` because that is what torch hands to CUDA.
    """
    K, N = qu.shape
    if K % PACK_FACTOR:
        raise ValueError(f"K={K} must be divisible by {PACK_FACTOR}")
    if qu.max(initial=0) > 15:
        raise ValueError("quantised values must fit in 4 bits")

    blocks = qu.reshape(K // PACK_FACTOR, PACK_FACTOR, N).astype(np.uint32)
    packed = np.zeros((K // PACK_FACTOR, N), dtype=np.uint32)
    for k_off in range(PACK_FACTOR):
        slot = int(SLOT_FOR_K[k_off])
        packed |= (blocks[:, k_off, :] & 0xF) << np.uint32(4 * slot)
    return packed.view(np.int32)


def unpack_nibbles(packed: np.ndarray, K: int) -> np.ndarray:
    """Inverse of :func:`pack_nibbles`; returns ``uint8[K, N]`` in ``[0, 15]``."""
    p = np.ascontiguousarray(packed).view(np.uint32)
    rows, N = p.shape
    if rows * PACK_FACTOR != K:
        raise ValueError(f"packed rows={rows} inconsistent with K={K}")

    out = np.empty((rows, PACK_FACTOR, N), dtype=np.uint8)
    for k_off in range(PACK_FACTOR):
        slot = int(SLOT_FOR_K[k_off])
        out[:, k_off, :] = ((p >> np.uint32(4 * slot)) & 0xF).astype(np.uint8)
    return out.reshape(K, N)


def dequantize(qw: QuantizedWeight) -> torch.Tensor:
    """Reconstruct the ``[K, N]`` fp16 weight the kernels are expected to model."""
    qu = unpack_nibbles(qw.qweight.cpu().numpy(), qw.K)
    q = torch.from_numpy(qu.astype(np.int16)).float() - ZERO_POINT
    scale = qw.scales.float().cpu().repeat_interleave(qw.group_size, dim=0)
    return (q * scale).to(torch.float16)


def reference_matmul(X: torch.Tensor, qw: QuantizedWeight) -> torch.Tensor:
    """fp32-accumulated reference for ``X @ dequant(qw)``.

    Accumulating in fp32 keeps the reference free of the fp16 rounding the
    kernels incur, so tests can attribute error to the kernel rather than to the
    baseline.
    """
    W = dequantize(qw).to(X.device).float()
    return (X.float() @ W).to(torch.float16)
