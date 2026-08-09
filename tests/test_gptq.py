"""GPTQ correctness and the property that justifies it.

Most of this file runs on CPU deliberately: the algorithm is pure linear algebra
and its central claim -- that it trades weight error for output error -- can be
checked without a GPU, so it stays testable on a laptop.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import nibblegemm as ng
from nibblegemm.gptq import HessianAccumulator, _inverse_cholesky, gptq_quantize


def make_layer(K, N, seed=0, device="cpu"):
    """A weight matrix and a correlated activation distribution.

    Correlation matters: with white activations the Hessian is near-diagonal and
    GPTQ degenerates toward RTN, so an uncorrelated test would pass while
    proving nothing.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    mix = torch.randn(K, K, generator=g) * 0.1 + torch.eye(K)
    X = torch.randn(2048, K, generator=g) @ mix
    X[:, : max(1, K // 32)] *= 6.0  # a few high-variance channels
    W = torch.randn(K, N, generator=g) * 0.02
    W[torch.rand(K, N, generator=g) < 0.002] *= 10  # weight outliers
    H = 2.0 * (X.t() @ X) / X.shape[0]
    return W.to(device), X.to(device), H.to(device)


def output_mse(W_ref, X, qw):
    return (X @ ng.dequantize(qw).float() - X @ W_ref).pow(2).mean().item()


def weight_mse(W_ref, qw):
    return (ng.dequantize(qw).float() - W_ref).pow(2).mean().item()


# ---------------------------------------------------------------------------
# Format
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,N,G", [(256, 64, 128), (512, 128, 64), (384, 32, 128)])
def test_output_format_matches_quantize(K, N, G):
    if K % G:
        pytest.skip("K must divide the group size")
    W, _, H = make_layer(K, N)
    qw = gptq_quantize(W, H, group_size=G)
    ref = ng.quantize(W, group_size=G)

    assert qw.qweight.shape == ref.qweight.shape == (K // 8, N)
    assert qw.scales.shape == ref.scales.shape == (K // G, N)
    assert qw.qweight.dtype == torch.int32
    assert qw.scales.dtype == torch.float16
    assert (qw.K, qw.N, qw.group_size) == (K, N, G)


def test_packed_values_are_four_bit():
    W, _, H = make_layer(256, 64)
    qw = gptq_quantize(W, H, group_size=128)
    nibbles = ng.unpack_nibbles(qw.qweight.numpy(), qw.K)
    assert nibbles.min() >= 0 and nibbles.max() <= 15


def test_scales_are_positive_and_finite():
    W, _, H = make_layer(256, 64)
    qw = gptq_quantize(W, H, group_size=128)
    s = qw.scales.float()
    assert torch.isfinite(s).all()
    assert (s > 0).all()


# ---------------------------------------------------------------------------
# The property GPTQ exists for
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("K,N", [(256, 64), (512, 128)])
def test_beats_rtn_on_output_error(K, N):
    """The objective that matters: error at the layer's output."""
    W, X, H = make_layer(K, N)
    rtn = ng.quantize(W, group_size=128)
    gptq = gptq_quantize(W, H, group_size=128)
    assert output_mse(W, X, gptq) < output_mse(W, X, rtn)


@pytest.mark.parametrize("K,N", [(256, 64), (512, 128)])
def test_is_worse_on_weight_error(K, N):
    """And the objective that does not.

    GPTQ moves weights *further* from their original values on purpose, because
    a compensating error in a later column cancels an earlier one at the output.
    Asserting this explicitly documents that a weight-MSE regression here is the
    algorithm working, not a bug -- and it is the same trap the clipping search
    in quant.py fell into from the other direction.
    """
    W, _, H = make_layer(K, N)
    rtn = ng.quantize(W, group_size=128)
    gptq = gptq_quantize(W, H, group_size=128)
    assert weight_mse(W, gptq) > weight_mse(W, rtn)


def test_identity_hessian_stays_close_to_rtn():
    """With uncorrelated activations there is nothing to compensate with.

    A diagonal Hessian means every column's error projects onto no other column,
    so GPTQ should land near RTN. This pins the algorithm against the degenerate
    case: a version that drifted far from RTN here would be redistributing error
    that the Hessian says is unrelated.
    """
    K, N = 256, 64
    W, _, _ = make_layer(K, N)
    H = torch.eye(K) * 2.0
    gptq = gptq_quantize(W, H, group_size=128)
    rtn = ng.quantize(W, group_size=128)
    rel = (ng.dequantize(gptq).float() - ng.dequantize(rtn).float()).abs().max() \
        / ng.dequantize(rtn).float().abs().max()
    assert rel < 0.5


# ---------------------------------------------------------------------------
# Numerical robustness
# ---------------------------------------------------------------------------
def test_dead_channels_are_handled():
    """Channels no calibration token activated make H singular."""
    K, N = 256, 64
    W, _, H = make_layer(K, N)
    H[10, :] = 0
    H[:, 10] = 0
    qw = gptq_quantize(W, H, group_size=128)
    assert torch.isfinite(qw.scales.float()).all()
    # A dead input channel contributes nothing, so it is zeroed before solving.
    assert ng.dequantize(qw)[10].abs().max().item() == 0.0


def test_singular_hessian_escalates_damping():
    """A rank-deficient Hessian must not silently produce garbage."""
    K = 128
    base = torch.randn(K, 4)
    H = base @ base.t()  # rank 4
    W, _, _ = make_layer(K, 32)
    qw = gptq_quantize(W, H, group_size=128)
    assert torch.isfinite(ng.dequantize(qw).float()).all()


def test_inverse_cholesky_is_upper_triangular():
    _, _, H = make_layer(128, 32)
    U = _inverse_cholesky(H, damp=0.01, retries=3)
    assert torch.allclose(U, torch.triu(U), atol=0)
    # UᵀU should reconstruct the (damped) inverse, so U must be well conditioned.
    assert torch.isfinite(U).all() and (torch.diag(U) > 0).all()


def test_rejects_shape_mismatch():
    W, _, H = make_layer(256, 64)
    with pytest.raises(ValueError):
        gptq_quantize(W, H[:128, :128], group_size=128)
    with pytest.raises(ValueError):
        gptq_quantize(W, H, group_size=100)  # K not divisible by group size


# ---------------------------------------------------------------------------
# Hessian accumulation
# ---------------------------------------------------------------------------
def test_hessian_accumulator_matches_direct_computation():
    """Batched accumulation must equal the one-shot result."""
    K = 32
    X = torch.randn(500, K)
    acc = HessianAccumulator(K, device="cpu")
    for start in range(0, 500, 64):
        acc.add(X[start:start + 64])
    expected = 2.0 * (X.t() @ X) / X.shape[0]
    torch.testing.assert_close(acc.H, expected, rtol=1e-4, atol=1e-4)


def test_hessian_accumulator_flattens_leading_dims():
    """Hooks see [batch, seq, features]; the accumulator must fold both."""
    acc = HessianAccumulator(16, device="cpu")
    acc.add(torch.randn(4, 8, 16))
    assert acc.n == 32


def test_hessian_accumulator_ignores_empty_batches():
    acc = HessianAccumulator(16, device="cpu")
    acc.add(torch.randn(0, 16))
    assert acc.n == 0
    assert torch.equal(acc.H, torch.zeros(16, 16))


# ---------------------------------------------------------------------------
# The kernel consumes GPTQ output unchanged
# ---------------------------------------------------------------------------
@pytest.mark.cuda
@pytest.mark.usefixtures("ext")
def test_kernel_runs_on_gptq_weights():
    """The whole point of choosing GPTQ over AWQ: the format does not change."""
    K, N = 512, 128
    W, _, H = make_layer(K, N)
    qw = gptq_quantize(W, H, group_size=128).to("cuda")
    X = torch.randn(1, K, dtype=torch.float16, device="cuda")

    ref = ng.reference_matmul(X, qw)
    got = ng.gemv(X, qw, version=4)
    rel = (got.float() - ref.float()).abs().max() / ref.float().abs().mean()
    # Same tolerance as test_gemv: fp16 accumulation over the reduction axis.
    assert rel < 5e-2
