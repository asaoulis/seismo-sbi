"""Pin the Instaseis source-depth datum offset.

Instaseis measures source depth from the *model's free surface*, which is not
always the sea-level datum catalogues use.  ``source_depth_offset_km`` corrects
that at the single point where a depth is handed to Instaseis, so catalogues,
prior boxes, conditioning vectors and posteriors all stay in one datum.

The default (0.0) must leave the depth unchanged, as every existing config expects.
"""
import numpy as np
import pytest

from seismo_sbi.simulators.instaseis.querier import InstaseisDBQuerier
from seismo_sbi.simulators.sources import GenericPointSource, GeneralMomentTensor, SourceLocation


def _querier(offset):
    """A querier with only the attributes _create_source_object needs (no DB open)."""
    q = object.__new__(InstaseisDBQuerier)
    q._dt = 0.5
    q.source_depth_offset_km = float(offset)
    q.stf_alignment = "peak"
    return q


def _source(depth_km):
    return GenericPointSource(
        SourceLocation(64.75, -17.2, depth_km, 0.0),
        GeneralMomentTensor(np.array([1e15, -1e15, 0.0, 0.0, 0.0, 0.0])),
    )


def test_default_offset_is_identity():
    """No offset configured -> depth passes through unchanged (legacy behaviour)."""
    src = _querier(0.0)._create_source_object(_source(6.5))
    assert src.depth_in_m == pytest.approx(6500.0)


def test_offset_shifts_depth_at_the_instaseis_boundary():
    """A 1 km datum offset puts a 6.5 km b.s.l. source at 7.5 km below the free surface."""
    src = _querier(1.0)._create_source_object(_source(6.5))
    assert src.depth_in_m == pytest.approx(7500.0)


def test_offset_is_not_applied_twice():
    """Two calls on the same querier must not accumulate the offset."""
    q = _querier(1.0)
    first = q._create_source_object(_source(6.5)).depth_in_m
    second = q._create_source_object(_source(6.5)).depth_in_m
    assert first == pytest.approx(second) == pytest.approx(7500.0)


def test_negative_guard_tests_the_post_offset_depth():
    """A depth above the catalogue datum is legal when the offset keeps it inside the model."""
    src = _querier(1.0)._create_source_object(_source(-0.5))
    assert src.depth_in_m == pytest.approx(500.0)


def test_negative_guard_still_rejects_depth_above_the_free_surface():
    """Post-offset depth above the free surface is still rejected, as before."""
    with pytest.raises(ValueError):
        _querier(1.0)._create_source_object(_source(-2.0))
    with pytest.raises(ValueError):
        _querier(0.0)._create_source_object(_source(-0.5))


def test_simulation_parameters_carries_the_offset():
    """The YAML `seismic_context.source_depth_offset_km` must reach SimulationParameters.

    It is a NamedTuple, so an unwired field raises rather than being ignored -- but a MISSING
    default would break every existing config, so pin both.
    """
    from seismo_sbi.sbi.types.parameters import SimulationParameters

    kw = dict(receivers=None, components="ZNE", seismogram_duration=200,
              syngine_address="x", sampling_rate=1.0, processing={})
    assert SimulationParameters(**kw).source_depth_offset_km == 0.0          # legacy default
    assert SimulationParameters(**kw, source_depth_offset_km=1.0).source_depth_offset_km == 1.0


def test_simulator_wrapper_forwards_the_offset(monkeypatch):
    """select_and_initialise_simulator must pass the offset to the Instaseis ensemble simulator.

    Without this the config key parses, reaches SimulationParameters, and is then silently
    dropped at simulator construction -- every source 1 km too shallow, with no error.
    """
    import seismo_sbi.sbi.simulator_wrapper as sw
    import seismo_sbi.simulators.registry as registry
    from seismo_sbi.sbi.types.parameters import SimulationParameters

    seen = {}

    class _Spy:
        def __init__(self, *a, **kw):
            seen.update(kw)

    monkeypatch.setattr(registry, "InstaseisEnsembleSimulator", _Spy)

    sp = SimulationParameters(receivers=None, components="ZNE", seismogram_duration=200,
                              syngine_address="ens", sampling_rate=1.0, processing={},
                              syngine_fiducial_address="fid", source_depth_offset_km=1.0)
    sw.GeneralSimulatorWrapper.select_and_initialise_simulator(
        object.__new__(sw.GeneralSimulatorWrapper), "instaseis_ensemble", sp)
    assert seen["source_depth_offset_km"] == 1.0
