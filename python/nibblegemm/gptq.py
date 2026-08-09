"""GPTQ: error-compensated INT4 quantisation.

Round-to-nearest costs this model 12.4% perplexity, and none of it is the
kernel's fault -- the CUDA path is bit-exact. The loss is entirely in *choosing*
the 4-bit values, and RTN chooses them the dumbest possible way: independently,
one weight at a time, as if the others did not exist.

They do exist. A layer's output is ``x @ W``, so what matters is not how far each
weight moved but how far the *output* moved, and an error introduced in one
weight can be partially cancelled by nudging the weights that have not been
quantised yet. That is GPTQ: quantise along the input dimension in order, and
after fixing each column, push its rounding error onto the remaining columns
weighted by the inverse Hessian of the layer's calibration activations.

Two properties make it the right choice here over AWQ:

* **The output format is unchanged.** GPTQ only produces different *values* for
  the same packed INT4 layout, so the kernel, the packing, and the tests all
  stay exactly as they are. AWQ instead rescales input channels and requires the
  reciprocal to be folded into an adjacent layer -- model surgery this repo has
  no reason to take on.
* **It attacks the metric that actually regressed.** The earlier clipping-search
  experiment failed because weight MSE is a proxy; GPTQ's objective is the
  layer's output error under the real activation distribution, which is the
  thing perplexity responds to.

The Hessian is ``H = 2 XᵀX`` over calibration activations, accumulated by
forward hooks so the activations themselves are never stored -- ``H`` is
``[K, K]`` regardless of how many tokens are pushed through it.

Reference: Frantar et al., "GPTQ: Accurate Post-Training Quantization for
Generative Pre-trained Transformers" (2023).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from .quant import QMAX, QMIN, ZERO_POINT, QuantizedWeight, pack_nibbles


class HessianAccumulator:
    """Running ``H = 2 XᵀX`` for one linear layer.

    Stored as a running mean rather than a raw sum so the magnitude does not
    depend on how many calibration tokens were used, which keeps the damping
    term (a fraction of ``mean(diag(H))``) meaningful across configurations.
    """

    def __init__(self, in_features: int, device, dtype=torch.float32):
        self.H = torch.zeros(in_features, in_features, device=device, dtype=dtype)
        self.n = 0

    @torch.no_grad()
    def add(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).to(self.H.dtype)
        rows = x.shape[0]
        if rows == 0:
            return
        self.H *= self.n / (self.n + rows)
        self.n += rows
        self.H += (2.0 / self.n) * (x.t() @ x)


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, group_size: int = 128,
                  damp: float = 0.01, max_damp_retries: int = 5) -> QuantizedWeight:
    """Quantise ``W`` (``[K, N]``, consumed as ``Y = X @ W``) using Hessian ``H`` (``[K, K]``).

    Columns are processed one group at a time so that a group's scale is chosen
    from weights that already carry the error pushed forward from every previous
    group. Blocking the error propagation at group granularity (rather than the
    reference implementation's independent block size) keeps the scale and the
    compensation consistent with each other; with the usual configuration the
    two block sizes coincide anyway.
    """
    K, N = W.shape
    if K % group_size:
        raise ValueError(f"K={K} must be divisible by group_size={group_size}")
    if H.shape != (K, K):
        raise ValueError(f"H must be [{K}, {K}], got {tuple(H.shape)}")

    device = W.device
    # Work in the [out, in] orientation the algorithm is written for: each
    # iteration fixes one input channel across all output channels at once.
    Wl = W.t().contiguous().float()  # [N, K]
    H = H.clone().float()

    # Channels no calibration token ever activated carry no information; the
    # Hessian is singular there. Zero them and give the diagonal a unit entry so
    # the factorisation stays well posed.
    dead = torch.diag(H) == 0
    if dead.any():
        H[dead, dead] = 1.0
        Wl[:, dead] = 0.0

    Hinv = _inverse_cholesky(H, damp, max_damp_retries)

    Q = torch.zeros_like(Wl)
    scales = torch.zeros(K // group_size, N, device=device, dtype=torch.float32)

    for i1 in range(0, K, group_size):
        i2 = i1 + group_size
        W1 = Wl[:, i1:i2].clone()          # [N, group_size]
        Hinv1 = Hinv[i1:i2, i1:i2]

        # Scale comes from the error-compensated weights, not the originals.
        scale = (W1.abs().amax(dim=1) / QMAX).clamp(min=1e-8)  # [N]
        scales[i1 // group_size] = scale
        inv_scale = 1.0 / scale

        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)

        for i in range(group_size):
            w = W1[:, i]
            q = torch.clamp(torch.round(w * inv_scale), QMIN, QMAX)
            Q1[:, i] = q
            # Residual, pre-divided by the diagonal so the update below is a
            # plain outer product.
            err = (w - q * scale) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err

        Q[:, i1:i2] = Q1
        if i2 < K:
            Wl[:, i2:] -= E1 @ Hinv[i1:i2, i2:]

    qu = (Q.t() + ZERO_POINT).to(torch.uint8).cpu().numpy()  # [K, N] in [0, 15]
    return QuantizedWeight(
        qweight=torch.from_numpy(pack_nibbles(qu)),
        scales=scales.to(torch.float16).cpu().contiguous(),
        group_size=group_size,
        K=K,
        N=N,
    )


def _inverse_cholesky(H: torch.Tensor, damp: float, retries: int) -> torch.Tensor:
    """Upper-triangular Cholesky factor of ``H⁻¹``, with escalating damping.

    The algorithm needs ``H⁻¹`` only through this factor: ``Hinv[i, i:]`` is
    exactly the coefficient vector for spreading column ``i``'s error over the
    columns after it. Real calibration Hessians are frequently near-singular
    (dead channels, correlated inputs), so damping is not optional -- and when
    the factorisation still fails, raising it and retrying is far better than
    returning silently wrong weights.

    The explicit ``triu`` is not cosmetic. LAPACK's ``potrf`` writes only the
    triangle it was asked for and leaves whatever was in the other half, so
    ``torch.linalg.cholesky(..., upper=True)`` can return a tensor with stale
    non-zeros below the diagonal. The algorithm above happens to read only
    on-or-above-diagonal entries, so this changes no result -- but a function
    that says "upper triangular" should return one, rather than leaving the next
    reader to rediscover the invariant.
    """
    K = H.shape[0]
    eye = torch.eye(K, device=H.device, dtype=H.dtype)
    mean_diag = torch.diag(H).mean().clamp(min=1e-8)

    for attempt in range(retries):
        scale = damp * (10 ** attempt)
        try:
            L = torch.linalg.cholesky(H + eye * (scale * mean_diag))
            Hinv = torch.cholesky_inverse(L)
            return torch.triu(torch.linalg.cholesky(Hinv, upper=True))
        except Exception:
            continue
    raise RuntimeError(
        f"Hessian factorisation failed after {retries} attempts up to damping "
        f"{damp * 10 ** (retries - 1):.3g}; the calibration set is probably degenerate"
    )


# ---------------------------------------------------------------------------
# Model-level driver
# ---------------------------------------------------------------------------
def find_decoder_blocks(model: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the repeated transformer blocks of a causal LM."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers", "gpt_neox.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, (nn.ModuleList, list)) and len(obj):
            return obj
    return None


def supported_linear(lin: nn.Linear, group_size: int) -> bool:
    K, N = lin.in_features, lin.out_features
    return K % group_size == 0 and N % 4 == 0 and K % 32 == 0


@torch.no_grad()
def quantize_model_gptq(model: nn.Module, calibration: Iterable[torch.Tensor],
                        group_size: int = 128, damp: float = 0.01,
                        make_quant_linear: Callable = None,
                        log: Callable[[str], None] = print) -> Tuple[int, List[str]]:
    """Quantise a causal LM in place with GPTQ, one decoder block at a time.

    Blocks are processed in order, and each block's linears are replaced before
    the next block is calibrated. That is deliberate: block *i+1* then sees the
    activations the quantised model will actually produce, rather than the fp16
    model's, so its Hessian describes the real input distribution and its error
    compensation partly absorbs the drift accumulated upstream.

    The cost is one calibration pass per block. Holding every layer's Hessian at
    once would need a single pass but tens of gigabytes on a real model, and
    would give up the sequential property above.
    """
    from .ops import QuantLinear
    make_quant_linear = make_quant_linear or (lambda qw, bias: QuantLinear(qw, bias))

    blocks = find_decoder_blocks(model)
    if blocks is None:
        raise RuntimeError("could not locate decoder blocks on this model")

    calibration = [c for c in calibration]
    device = next(model.parameters()).device
    swapped, skipped = 0, []

    for b, block in enumerate(blocks):
        targets: Dict[str, nn.Linear] = {
            name: mod for name, mod in block.named_modules() if isinstance(mod, nn.Linear)
        }
        for name, lin in list(targets.items()):
            if not supported_linear(lin, group_size):
                skipped.append(f"block{b}.{name} [{lin.in_features}x{lin.out_features}]")
                targets.pop(name)
        if not targets:
            continue

        accs = {n: HessianAccumulator(l.in_features, device) for n, l in targets.items()}
        handles = [
            lin.register_forward_pre_hook(_make_hook(accs[name]))
            for name, lin in targets.items()
        ]
        try:
            for batch in calibration:
                model(batch.to(device))
        finally:
            for h in handles:
                h.remove()

        for name, lin in targets.items():
            W = lin.weight.data.t().contiguous()  # [out, in] -> [in, out]
            qw = gptq_quantize(W, accs[name].H, group_size=group_size, damp=damp)
            bias = lin.bias.data.clone() if lin.bias is not None else None
            _set_submodule(block, name, make_quant_linear(qw.to(device), bias))
            swapped += 1
        del accs
        torch.cuda.empty_cache()
        log(f"  block {b + 1}/{len(blocks)}: {len(targets)} layers quantised")

    return swapped, skipped


def _make_hook(acc: HessianAccumulator):
    def hook(_module, args):
        if args and isinstance(args[0], torch.Tensor):
            acc.add(args[0].detach())
        return None
    return hook


def _set_submodule(root: nn.Module, dotted: str, value: nn.Module) -> None:
    parts = dotted.split(".")
    for p in parts[:-1]:
        root = getattr(root, p)
    setattr(root, parts[-1], value)
