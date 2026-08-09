"""Correctness of the decode-regime GEMV ladder (versions 0..4).

Every rung is supposed to be correct, so all of them are checked against the same
fp32-accumulated reference and against each other.
"""
from __future__ import annotations

import pytest
import torch

import nibblegemm as ng

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("ext")]

VERSIONS = [0, 1, 2, 3, 4]

# N=132 is not a power of two and not a multiple of the 512-column v4 block tile,
# while still satisfying the kernel's N % 4 == 0 rule for 128-bit weight loads.
SHAPES = [(512, 256), (256, 132)]

# v2 accumulates a whole 128-wide group in half2 before flushing to fp32; v3/v4
# flush every FLUSH_WORDS=4 packed words, i.e. every 32 values. The reference
# accumulates everything in fp32. Accumulating W values in fp16 costs about
# sqrt(W) * 2**-11 relative on the windowed partial (~5e-3 for W=128), and the
# metric divides a max over outputs by a mean magnitude, which for a Gaussian dot
# product inflates it a further ~4x -- so the expected number is near 1e-2.
# 5e-2 is what scripts/smoke.py uses and leaves roughly 4x headroom; a genuine
# indexing or decode bug shows up at O(1) on this metric, not at 6e-2.
REL_TOL = 5e-2


def make_case(K: int, N: int, M: int, group_size: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(K, N, generator=g) * 0.02
    qw = ng.quantize(W, group_size=group_size).to("cuda")
    X = torch.randn(M, K, generator=g).to(device="cuda", dtype=torch.float16)
    return X, qw


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("M", [1, 2, 4, 8])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("K,N", SHAPES)
def test_gemv_matches_reference(version, M, group_size, K, N, rel_err):
    X, qw = make_case(K, N, M, group_size)
    y = ng.gemv(X, qw, version=version)
    assert y.shape == (M, N)
    assert y.dtype == torch.float16
    assert rel_err(y, ng.reference_matmul(X, qw)) < REL_TOL


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("K,N", SHAPES)
def test_gemv_versions_agree(M, group_size, K, N, rel_err):
    X, qw = make_case(K, N, M, group_size, seed=1)
    # v0 is itself an fp32-accumulating scalar-decode reference, so the gap to any
    # other rung is exactly the fp16 windowing REL_TOL is sized for.
    base = ng.gemv(X, qw, version=0)
    for version in VERSIONS[1:]:
        assert rel_err(ng.gemv(X, qw, version=version), base) < REL_TOL, f"v{version} vs v0"


# 1024/64 = 16 groups, so splits of 1, 4 and 16 all survive the "trim empty
# splits" step in gemv_w4a16 and genuinely partition the reduction differently.
SPLIT_K, SPLIT_N, SPLIT_G = 1024, 256, 64

# Both sides accumulate the split partials in fp32, so the only real difference is
# fp32 summation order (~1e-7 relative) followed by the one fp16 store: at most
# 1 ULP = 2**-10 relative on the largest output, and max|y| runs ~5x mean|y| here,
# giving ~5e-3. 2e-2 is 4x that. A dropped or double-counted split would move the
# result by ~1/sqrt(splits) of its own magnitude -- O(0.25), not O(0.02).
SPLIT_TOL = 2e-2


@pytest.mark.parametrize("M", [1, 4, 8])
@pytest.mark.parametrize("splits", [1, 4, 16])
def test_gemv_splitk_is_invariant(M, splits, rel_err):
    X, qw = make_case(SPLIT_K, SPLIT_N, M, SPLIT_G, seed=2)
    y = ng.gemv(X, qw, version=4, splits=splits)
    single = ng.gemv(X, qw, version=3)  # v3 is v4 pinned to a single split
    assert rel_err(y, single) < SPLIT_TOL
    assert rel_err(y, ng.reference_matmul(X, qw)) < REL_TOL


def test_v3_is_v4_with_one_split():
    """Not a tolerance: version 3 and ``version=4, splits=1`` launch the same
    template with the same arguments and the same nullptr workspace."""
    X, qw = make_case(SPLIT_K, SPLIT_N, 4, SPLIT_G, seed=3)
    assert torch.equal(ng.gemv(X, qw, version=4, splits=1), ng.gemv(X, qw, version=3))


@pytest.mark.parametrize("version", [3, 4])
def test_gemv_rejects_m_above_8(version):
    # Only the register-blocked rungs are templated on M and so guard it; v0..v2
    # map one thread per output and have no such limit.
    X, qw = make_case(256, 132, 9, 128, seed=4)
    with pytest.raises(RuntimeError):
        ng.gemv(X, qw, version=version)


def test_matmul_dispatches_to_gemv_below_the_crossover():
    X, qw = make_case(512, 256, ng.DECODE_MAX_M, 128, seed=5)
    assert torch.equal(ng.matmul(X, qw), ng.gemv(X, qw, version=4))


def test_matmul_rejects_k_mismatch():
    _, qw = make_case(512, 256, 1, 128, seed=6)
    X = torch.randn(1, 256, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        ng.matmul(X, qw)
