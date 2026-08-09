"""Shared pytest setup.

The repo is not installed as a package, so ``python/`` goes on ``sys.path`` here
rather than relying on the caller's PYTHONPATH.

Tests that touch a kernel carry ``@pytest.mark.cuda`` and are skipped wholesale
on a machine with no GPU. tests/test_pack.py is deliberately unmarked: it pins
the on-disk nibble layout that csrc/ decodes, and that has to be checkable on the
development machine.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_PYTHON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
if _PYTHON not in sys.path:
    sys.path.insert(0, _PYTHON)

import nibblegemm as ng  # noqa: E402

HAS_CUDA = torch.cuda.is_available()


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: requires a CUDA device")


def pytest_collection_modifyitems(config, items):
    if HAS_CUDA:
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _exact_fp32_reference():
    """Keep the fp32 reference matmuls out of TF32.

    ``reference_matmul`` is the ground truth every tolerance in this suite is
    stated against; letting cuBLAS silently drop it to a 10-bit mantissa would
    fold the reference's own error into those numbers.
    """
    if HAS_CUDA:
        torch.backends.cuda.matmul.allow_tf32 = False
    yield


@pytest.fixture(scope="session")
def ext():
    """Build the CUDA extension once per session.

    The JIT build takes minutes; paying it in a session fixture keeps a build
    failure to one clear error instead of one per test.
    """
    return ng.extension()


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def rel_err():
    """``max|y - ref| / mean|ref|`` -- the metric every tolerance below is in.

    Normalising by the mean magnitude of the whole tensor rather than per element
    keeps the metric finite where an individual output lands near zero, which for
    a random dot product it regularly does.
    """

    def _rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
        y, ref = y.float(), ref.float()
        return ((y - ref).abs().max() / ref.abs().mean()).item()

    return _rel_err
