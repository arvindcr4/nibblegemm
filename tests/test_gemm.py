"""Correctness of the prefill-regime tensor-core GEMM.

The interesting cases are the edges. The kernel tiles 64x64 with masked activation
loads and a masked epilogue, so M and N are chosen to land on, just past, and far
past a tile boundary.
"""
from __future__ import annotations

import pytest
import torch

import nibblegemm as ng

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("ext")]

K = 256  # the 64-deep block tile requires K % 64 == 0

# 1 and 17 are a fraction of a tile; 64 and 128 are exact multiples; 65 is one row
# into a second tile, leaving 63 masked rows; 1000 is 15 tiles plus a 40-row
# remainder.
M_VALUES = [1, 17, 64, 65, 128, 1000]

# 128 is two whole column tiles; 132 leaves 60 masked columns in the last one and
# is still a multiple of 4.
N_VALUES = [128, 132]

# The shared B tile holds exactly the fp16 values dequantize() produces, and both
# the wmma accumulators and the reference sum in fp32 -- so only fp32 summation
# order (~1e-6 relative) and the final fp16 store differ. That store is worth at
# most 1 ULP, 2**-10 relative, on the largest output, and max|ref| runs ~5.5x
# mean|ref| for a Gaussian dot product, giving ~5e-3. 3e-2 is ~5x that and still
# far below the O(1) error a wrong edge mask produces: an unmasked row reads a
# neighbouring row's data, a wrong mask writes zeros.
REL_TOL = 3e-2


def make_case(M: int, N: int, group_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(K, N, generator=g) * 0.02
    qw = ng.quantize(W, group_size=group_size).to("cuda")
    X = torch.randn(M, K, generator=g).to(device="cuda", dtype=torch.float16)
    return X, qw


@pytest.mark.parametrize("M", M_VALUES)
@pytest.mark.parametrize("N", N_VALUES)
@pytest.mark.parametrize("group_size", [64, 128])
def test_gemm_matches_reference(M, N, group_size, rel_err):
    X, qw = make_case(M, N, group_size)
    y = ng.gemm(X, qw)
    assert y.shape == (M, N)
    assert y.dtype == torch.float16
    assert rel_err(y, ng.reference_matmul(X, qw)) < REL_TOL


@pytest.mark.parametrize("M", [16, 64, 512])
def test_matmul_dispatches_to_gemm_above_the_crossover(M):
    X, qw = make_case(M, 128, 128, seed=1)
    assert M > ng.DECODE_MAX_M
    assert torch.equal(ng.matmul(X, qw), ng.gemm(X, qw))
