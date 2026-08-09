"""On-disk nibble format: the pack/unpack involution and the slot interleave.

No ``cuda`` marker -- this file pins the layout that every kernel in csrc/
decodes, so it must run on any machine.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import nibblegemm as ng
from nibblegemm.quant import PACK_FACTOR, SLOT_FOR_K

#: Spelled out as a literal on purpose: asserting against ``quant.SLOT_FOR_K``
#: alone would only prove the module agrees with itself.
EXPECTED_SLOT_FOR_K = [0, 4, 1, 5, 2, 6, 3, 7]


def test_slot_table_is_the_documented_permutation():
    assert list(SLOT_FOR_K) == EXPECTED_SLOT_FOR_K


def test_slot_table_matches_the_cuda_scalar_decode():
    """``dequant_scalar`` in csrc/dequant.cuh recomputes this mapping in C++."""
    cuda_slot = [(4 + (k >> 1)) if (k & 1) else (k >> 1) for k in range(PACK_FACTOR)]
    assert cuda_slot == EXPECTED_SLOT_FOR_K


def test_nibble_slot_assignment_is_pinned():
    """Logical offset ``k`` must land in nibble slot ``EXPECTED_SLOT_FOR_K[k]``.

    The four lop3 ops emit slots in the order 0,4,1,5,2,6,3,7 and rely on this
    permutation to hand back ascending ``k`` with no shuffle. A layout change
    here is silent corruption there, with no shape or dtype to catch it.
    """
    qu = np.arange(PACK_FACTOR, dtype=np.uint8).reshape(PACK_FACTOR, 1)
    word = int(ng.pack_nibbles(qu).view(np.uint32)[0, 0])
    for k, slot in enumerate(EXPECTED_SLOT_FOR_K):
        assert (word >> (4 * slot)) & 0xF == k
    # Same statement written out as the whole word, nibble 7 down to nibble 0.
    assert word == 0x75316420


@pytest.mark.parametrize("shape", [(8, 1), (8, 4), (256, 64), (128, 130), (1024, 3), (2048, 1)])
def test_pack_unpack_is_an_involution(shape):
    K, N = shape
    qu = np.random.default_rng(K * 31 + N).integers(0, 16, size=shape, dtype=np.uint8)
    packed = ng.pack_nibbles(qu)
    assert packed.dtype == np.int32
    assert packed.shape == (K // PACK_FACTOR, N)
    assert np.array_equal(ng.unpack_nibbles(packed, K), qu)


def test_pack_rejects_values_above_15():
    qu = np.zeros((8, 4), dtype=np.uint8)
    qu[5, 2] = 16
    with pytest.raises(ValueError):
        ng.pack_nibbles(qu)


def test_pack_rejects_k_not_divisible_by_8():
    with pytest.raises(ValueError):
        ng.pack_nibbles(np.zeros((12, 4), dtype=np.uint8))


def test_unpack_rejects_inconsistent_k():
    packed = ng.pack_nibbles(np.zeros((16, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        ng.unpack_nibbles(packed, 24)


@pytest.mark.parametrize("K,group_size", [(192, 128), (100, 64), (8, 128), (130, 128)])
def test_quantize_rejects_k_not_divisible_by_group(K, group_size):
    with pytest.raises(ValueError):
        ng.quantize(torch.randn(K, 8), group_size=group_size)


def test_quantize_rejects_non_2d_weight():
    with pytest.raises(ValueError):
        ng.quantize(torch.randn(4, 128, 8), group_size=128)
