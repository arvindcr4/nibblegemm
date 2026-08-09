"""QuantLinear as a drop-in nn.Linear replacement.

The point of these assertions is attribution. A W4A16 layer is *supposed* to be
wrong relative to its fp16 original -- INT4 with group-128 symmetric scales puts
several percent on the output -- so an end-to-end tolerance against the fp16
layer would pass almost any kernel. Instead the kernel is compared against exact
math on the same INT4 weights, and that gap is required to be small next to the
gap the quantisation itself opens.
"""
from __future__ import annotations

import pytest
import torch

import nibblegemm as ng

pytestmark = [pytest.mark.cuda, pytest.mark.usefixtures("ext")]

IN, OUT = 256, 128  # IN % group_size == 0 and IN % 64 == 0; OUT % 4 == 0

# Same bound as tests/test_gemv.py: the fp16 accumulation windows in the gemv
# ladder dominate, and the gemm path is tighter still.
REL_TOL = 5e-2


def make_layer(bias: bool, group_size: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    linear = torch.nn.Linear(IN, OUT, bias=bias).cuda().half()
    return linear, ng.QuantLinear.from_linear(linear, group_size=group_size)


def quantized_weight(ql: ng.QuantLinear) -> ng.QuantizedWeight:
    """The buffers QuantLinear holds, rewrapped so reference code can use them."""
    return ng.QuantizedWeight(ql.qweight, ql.scales, ql.group_size, ql.K, ql.N)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M", [1, 8, 64])
@pytest.mark.parametrize("group_size", [64, 128])
def test_error_belongs_to_quantisation_not_to_the_kernel(bias, M, group_size, rel_err):
    linear, ql = make_layer(bias, group_size)
    x = torch.randn(M, IN, device="cuda", dtype=torch.float16)

    y = ql(x)
    assert y.shape == (M, OUT)
    assert y.dtype == torch.float16

    exact_int4 = ng.reference_matmul(x, quantized_weight(ql))
    if bias:
        exact_int4 = exact_int4 + ql.bias
    exact_fp16 = linear(x)

    kernel_err = (y.float() - exact_int4.float()).abs().mean().item()
    quant_err = (exact_int4.float() - exact_fp16.float()).abs().mean().item()

    # INT4 quantisation of a kaiming-uniform weight puts ~10% mean relative error
    # on the output; the kernel's fp16 accumulation windows put well under 0.1%,
    # so the ratio sits near 0.01. 0.25 is a >20x margin that still fails loudly
    # the moment the kernel starts being a material contributor.
    assert quant_err > 0.0
    assert kernel_err < 0.25 * quant_err
    assert rel_err(y, exact_int4) < REL_TOL


@pytest.mark.parametrize("shape", [(1, 4, IN), (2, 8, IN), (3, 1, IN), (2, 64, IN)])
def test_3d_input_is_flattened_and_restored(shape, rel_err):
    """(batch, seq, hidden) collapses to a 2-D matmul; only the last dim changes."""
    _, ql = make_layer(bias=True, seed=1)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)

    y = ql(x)
    assert y.shape == (*shape[:-1], OUT)

    flat = ng.reference_matmul(x.reshape(-1, IN), quantized_weight(ql)) + ql.bias
    assert rel_err(y.reshape(-1, OUT), flat) < REL_TOL


def test_bias_is_added_exactly():
    _, ql = make_layer(bias=True, seed=2)
    assert ql.bias is not None
    x = torch.randn(4, IN, device="cuda", dtype=torch.float16)
    # Not a tolerance: forward() is exactly matmul followed by an fp16 add, and
    # the kernel is deterministic, so the two sides are the same instructions.
    assert torch.equal(ql(x), ng.matmul(x, quantized_weight(ql)) + ql.bias)


def test_no_bias_is_a_plain_matmul():
    _, ql = make_layer(bias=False, seed=3)
    assert ql.bias is None
    x = torch.randn(4, IN, device="cuda", dtype=torch.float16)
    assert torch.equal(ql(x), ng.matmul(x, quantized_weight(ql)))


def test_from_linear_transposes_the_weight():
    """nn.Linear stores [out, in]; the kernels consume [in, out]."""
    linear, ql = make_layer(bias=False, seed=4)
    assert (ql.K, ql.N) == (IN, OUT)

    approx = ng.dequantize(quantized_weight(ql)).float()
    truth = linear.weight.data.t().float().cpu()
    assert approx.shape == truth.shape
    # Per-element INT4 error on a kaiming-uniform weight is ~7% of mean|w|; a
    # permuted weight would be ~110%, since uncorrelated entries differ by more
    # than either one's magnitude. 0.25 sits between the two by a wide margin.
    assert (approx - truth).abs().mean() < 0.25 * truth.abs().mean()
