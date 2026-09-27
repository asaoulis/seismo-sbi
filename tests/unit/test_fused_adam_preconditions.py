"""Fused-AdamW precondition guard.

torch validates the `fused=True` preconditions LAZILY — inside the first
`optimizer.step()`, not in the constructor — so a try/except around `AdamW(...,
fused=True)` catches nothing. Job 1342786 (`--arch pno`, ml_perf.fused_adam: true) died
this way after ~70 minutes of sim-cache preload:

    RuntimeError: `fused=True` requires all the params to be floating point Tensors of
    supported devices: [...] but torch.complex64 and cuda

because the pno encoder's SpectralConv1d spectral weights are complex64. These tests pin
the eager precondition check that makes fused_adam safe for every architecture.

Pure CPU, no GPU required.
"""

import pytest
import torch

from seismo_sbi.sbi.compression.ML.seismogram_transformer import fused_adam_supported

pytestmark = pytest.mark.unit


class _FakeParam:
    """Stand-in for a Parameter so the CUDA branch is testable without a GPU."""

    def __init__(self, is_cuda, complex_=False):
        self._is_cuda = is_cuda
        self._complex = complex_

    @property
    def is_cuda(self):
        return self._is_cuda

    def is_complex(self):
        return self._complex


def test_complex_params_block_fused_even_on_cuda():
    """THE REGRESSION: complex64 on CUDA must NOT take the fused path."""
    params = [_FakeParam(is_cuda=True), _FakeParam(is_cuda=True, complex_=True)]
    assert fused_adam_supported(params) is False


def test_all_real_cuda_params_allow_fused():
    params = [_FakeParam(is_cuda=True), _FakeParam(is_cuda=True)]
    assert fused_adam_supported(params) is True


def test_cpu_params_block_fused():
    """Pre-existing device guard must survive the dtype addition."""
    assert fused_adam_supported([_FakeParam(is_cuda=False)]) is False
    # Mixed device is also unsafe.
    assert fused_adam_supported([_FakeParam(True), _FakeParam(False)]) is False


def test_empty_params_are_not_fused():
    assert fused_adam_supported([]) is False


def test_accepts_a_generator_not_just_a_list():
    """self.parameters() is a generator — a naive impl would consume it twice and lie."""
    gen = (_FakeParam(is_cuda=True) for _ in range(3))
    assert fused_adam_supported(gen) is True
    gen2 = (_FakeParam(is_cuda=True, complex_=(i == 2)) for i in range(3))
    assert fused_adam_supported(gen2) is False


def test_real_torch_complex_parameter_is_detected():
    """Guard against the FakeParam abstraction drifting from real torch semantics."""
    real = torch.nn.Parameter(torch.randn(4))
    cplx = torch.nn.Parameter(torch.randn(4, dtype=torch.complex64))
    assert real.is_complex() is False
    assert cplx.is_complex() is True
    # On CPU both are blocked by the device guard; the dtype guard is what matters on GPU.
    assert fused_adam_supported([real, cplx]) is False


def test_unfused_adamw_actually_steps_with_complex_params():
    """The fallback path must genuinely work — this is what the crashed job needed."""
    cplx = torch.nn.Parameter(torch.randn(4, dtype=torch.complex64))
    opt = torch.optim.AdamW([cplx], lr=1e-3)          # unfused, as the guard forces
    (cplx.abs() ** 2).sum().backward()
    opt.step()                                        # would raise if fused=True were used
    assert torch.isfinite(cplx.abs()).all()
