#!/usr/bin/env python3
"""The per-worker open-Instaseis-DB cache cap is a MEMORY BUDGET and must be honoured.

Each entry in ``ensemble._QUERIER_CACHE`` is an open Instaseis DB handle costing ~55 MB
resident (measured; independent of instaseis ``buffer_size_in_mb``, so it is DB metadata rather
than the Green's-function buffer). The cache is per worker PROCESS, so dataset generation costs
``n_workers * min(cap, n_members) * 55 MB``.

The regression these tests pin: ``_ensure_querier_cache_capacity`` used to grow the cap to the
ensemble size UNCONDITIONALLY, silently overriding an explicit ``SEISMO_QUERIER_CACHE_MAXSIZE``.
With a 62-member Mode-A/B ensemble at 60 joblib workers that is ~206 GB, which OOM-killed a 500k
dataset generation at 47% (the loky worker died on SIGKILL with no Python traceback).

Honouring the cap costs only cache misses — one ``instaseis.open_db`` each — and can never change
a simulated waveform, so it is always safe to impose.
"""
import importlib
import os

import pytest


def _reload_ensemble(monkeypatch, cap=None):
    """Re-import the module so the env is read fresh (the cap is resolved at import)."""
    if cap is None:
        monkeypatch.delenv("SEISMO_QUERIER_CACHE_MAXSIZE", raising=False)
    else:
        monkeypatch.setenv("SEISMO_QUERIER_CACHE_MAXSIZE", str(cap))
    import seismo_sbi.instaseis_simulator.ensemble as ens
    return importlib.reload(ens)


def test_an_explicit_cap_survives_a_larger_ensemble(monkeypatch):
    """The OOM regression: a 61-member ensemble must NOT reinstate an unbounded cache."""
    ens = _reload_ensemble(monkeypatch, cap=20)
    assert ens._QUERIER_CACHE_MAXSIZE == 20
    assert ens._QUERIER_CACHE_MAXSIZE_IS_EXPLICIT is True

    ens._ensure_querier_cache_capacity(61)     # brustle ensemble
    ens._ensure_querier_cache_capacity(31)     # mode_a
    ens._ensure_querier_cache_capacity(31)     # mode_b
    assert ens._QUERIER_CACHE_MAXSIZE == 20, (
        "an explicit SEISMO_QUERIER_CACHE_MAXSIZE is a memory budget and must be authoritative"
    )


def test_the_default_still_auto_grows_to_the_ensemble(monkeypatch):
    """Behaviour with the env UNSET is unchanged — no regression for existing callers."""
    ens = _reload_ensemble(monkeypatch, cap=None)
    assert ens._QUERIER_CACHE_MAXSIZE == 64          # documented default
    assert ens._QUERIER_CACHE_MAXSIZE_IS_EXPLICIT is False

    ens._ensure_querier_cache_capacity(120)
    assert ens._QUERIER_CACHE_MAXSIZE == 120, "the default must still grow to fit one ensemble"

    ens._ensure_querier_cache_capacity(10)           # never shrinks
    assert ens._QUERIER_CACHE_MAXSIZE == 120


def test_an_explicit_cap_below_the_default_is_respected(monkeypatch):
    """A cap smaller than the 64 default must not be silently raised to it."""
    ens = _reload_ensemble(monkeypatch, cap=8)
    assert ens._QUERIER_CACHE_MAXSIZE == 8
    ens._ensure_querier_cache_capacity(64)
    assert ens._QUERIER_CACHE_MAXSIZE == 8


def test_the_cap_is_a_config_field_so_it_can_be_set_per_run():
    """`seismic_context.querier_cache_maxsize` must parse (SimulationParameters is a NamedTuple,
    so an unknown key is a hard TypeError at config-parse time)."""
    from seismo_sbi.sbi.types.parameters import SimulationParameters

    assert "querier_cache_maxsize" in SimulationParameters._fields
    assert SimulationParameters._field_defaults["querier_cache_maxsize"] is None


@pytest.fixture(autouse=True)
def _restore_module_state():
    """Leave the imported module in its default state for other tests in the session."""
    yield
    os.environ.pop("SEISMO_QUERIER_CACHE_MAXSIZE", None)
    import seismo_sbi.instaseis_simulator.ensemble as ens
    importlib.reload(ens)
