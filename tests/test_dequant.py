"""Bit-exactness of the lop3 INT4 -> fp16 decode.

Everything here demands ``torch.equal``, never a tolerance. Both device paths and
the NumPy reference compute ``(nibble - 8) * scale`` with a single round to fp16:

  * the lop3 path builds ``1024 + n`` (or ``1024 + 16n``) inside fp16's
    exactly-representable integer range and removes the bias with one hsub2/hfma2,
    so ``n - 8`` is exact before the scale multiply;
  * the scalar path and the reference form ``(n - 8) * scale`` in fp32, where the
    product of a 4-bit integer and an 11-bit significand is exact.

So all three round the *same* exact real number once. Any difference at all is a
wrong magic constant, a wrong mask, or a wrong slot -- not noise. A tolerance
here would pass a decode that is quietly off by a whole nibble in one slot.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import nibblegemm as ng

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("ext")]


def sweep_nibbles(K: int = 128, extra_cols: int = 116, seed: int = 0) -> np.ndarray:
    """``[K, N]`` uint8 hitting every nibble value in every one of the 8 slots.

    Three column blocks:

    * ``0..127``   -- value ``v`` alone at slot ``s`` with 0xF in the other seven
      slots, for all 128 ``(s, v)`` pairs. Catches a mask that bleeds between
      slots, which a zero background would hide.
    * ``128..143`` -- all eight slots set to ``v``. Catches a wrong magic or bias
      constant independently of the slot permutation.
    * the rest     -- random words, for interactions the first two blocks miss.
    """
    k_off = np.arange(K) % 8
    slot, value = np.divmod(np.arange(128), 16)
    isolated = np.where(k_off[:, None] == slot[None, :], value[None, :], 15)
    uniform = np.tile(np.arange(16), (K, 1))
    rand = np.random.default_rng(seed).integers(0, 16, size=(K, extra_cols))
    return np.concatenate([isolated, uniform, rand], axis=1).astype(np.uint8)


def make_qw(qu: np.ndarray, scales: np.ndarray, group_size: int) -> ng.QuantizedWeight:
    """Build a QuantizedWeight straight from chosen nibbles, bypassing quantize().

    quantize() can only ever produce the nibbles some float happened to round to;
    these tests need to name the nibbles.
    """
    K, N = qu.shape
    return ng.QuantizedWeight(
        qweight=torch.from_numpy(ng.pack_nibbles(qu)),
        scales=torch.from_numpy(np.ascontiguousarray(scales, dtype=np.float32)).to(torch.float16),
        group_size=group_size,
        K=K,
        N=N,
    ).to("cuda")


def assert_all_three_agree(qw: ng.QuantizedWeight) -> torch.Tensor:
    fast = ng.dequant(qw, fast=True)
    slow = ng.dequant(qw, fast=False)
    ref = ng.dequantize(qw).to(fast.device)
    assert fast.shape == (qw.K, qw.N) and fast.dtype == torch.float16
    assert torch.equal(fast, slow), "lop3 decode disagrees with the scalar decode"
    assert torch.equal(fast, ref), "lop3 decode disagrees with the numpy reference"
    return fast


@pytest.mark.parametrize("group_size", [64, 128])
def test_decode_bit_exact_unit_scales(group_size):
    qu = sweep_nibbles(K=128)
    scales = np.ones((qu.shape[0] // group_size, qu.shape[1]), dtype=np.float32)
    fast = assert_all_three_agree(make_qw(qu, scales, group_size))
    # Not vacuous: with unit scales the sweep really does span the full codebook.
    assert fast.min().item() == -8.0
    assert fast.max().item() == 7.0


@pytest.mark.parametrize("group_size", [64, 128])
def test_decode_bit_exact_varied_scales(group_size):
    qu = sweep_nibbles(K=256, seed=1)
    rows, N = qu.shape[0] // group_size, qu.shape[1]
    rng = np.random.default_rng(2)
    # Scales span 2**-10 .. 2**10, so products span 2**-10 .. 2**13: five orders of
    # magnitude, all inside fp16's *normal* range. Straying into subnormals or
    # overflow would compare two paths' saturation behaviour rather than the
    # decode, which is not what this test is for.
    scales = (2.0 ** rng.uniform(-10.0, 10.0, size=(rows, N))).astype(np.float32)
    scales[0, :4] = [2.0**-10, 2.0**10, 1.0, 2.0**-3]  # the extremes, pinned
    assert_all_three_agree(make_qw(qu, scales, group_size))


@pytest.mark.parametrize("group_size", [64, 128])
def test_decoded_values_stay_within_the_codebook_range(group_size):
    qu = sweep_nibbles(K=256, seed=3)
    rows, N = qu.shape[0] // group_size, qu.shape[1]
    rng = np.random.default_rng(4)
    scales = (2.0 ** rng.uniform(-6.0, 6.0, size=(rows, N))).astype(np.float32)
    qw = make_qw(qu, scales, group_size)

    out = ng.dequant(qw, fast=True).float().cpu()
    s = qw.scales.float().cpu().repeat_interleave(group_size, dim=0)
    # (nibble - 8) is exact, so the only slack on the bound is the single
    # round-to-nearest of the product: half an fp16 ULP, i.e. 2**-11 relative.
    slack = 1.0 + 2.0**-11
    assert torch.all(out <= 7.0 * s * slack)
    assert torch.all(out >= -8.0 * s * slack)


@pytest.mark.parametrize("group_size", [64, 128])
def test_scales_are_applied_per_group(group_size):
    """A group-row off-by-one survives every fast-vs-scalar check (both kernels
    index the scales identically) and is only caught against the reference."""
    qu = sweep_nibbles(K=512, extra_cols=4, seed=5)
    rows, N = qu.shape[0] // group_size, qu.shape[1]
    # Distinct per-group magnitudes: reading the neighbouring group's scale moves
    # a whole row by 2x, far outside anything rounding could excuse.
    scales = np.repeat((2.0 ** np.arange(rows, dtype=np.float32))[:, None], N, axis=1)
    assert_all_three_agree(make_qw(qu, scales, group_size))
