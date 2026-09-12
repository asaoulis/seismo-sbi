"""Dispatching a ``simulation_type`` to a forward-model builder."""

import pytest

from seismo_sbi.simulators import registry


@pytest.fixture
def restore_registry():
    original = dict(registry.SIMULATOR_REGISTRY)
    yield
    registry.SIMULATOR_REGISTRY.clear()
    registry.SIMULATOR_REGISTRY.update(original)


def test_every_configurable_simulation_type_has_a_builder():
    assert sorted(registry.SIMULATOR_REGISTRY) == [
        "cps", "cps_covariance", "cps_multi", "cps_precomputed",
        "instaseis", "instaseis_ensemble", "instaseis_multi_ensemble", "kernel",
    ]


def test_an_unknown_simulation_type_lists_the_known_ones():
    with pytest.raises(NotImplementedError, match="instaseis_ensemble"):
        registry.build_simulator(("specfem3d", None), None)


def test_a_registered_builder_is_dispatched_to(restore_registry):
    seen = {}

    def builder(simulation_parameters, simulator_config, pp_effects, data_flattening):
        seen.update(config=simulator_config, effects=pp_effects, flatten=data_flattening)
        return "my simulator"

    registry.register_simulator("specfem3d", builder)
    assert registry.build_simulator(("specfem3d", 7), "params") == "my simulator"
    assert seen == {"config": ("specfem3d", 7), "effects": [], "flatten": None}


def test_kernel_builder_reads_the_kernels_from_the_config_payload():
    from seismo_sbi.simulators.receivers import Receiver, Receivers

    class _Parameters:
        components = ["Z"]
        receivers = Receivers(receivers=[Receiver(0.0, 0.0, "XX", "STA1", ["Z"])])
        seismogram_duration = 40
        processing = {"sampling_rate": 1.0}

    simulator = registry.build_simulator(("kernel", None), _Parameters())
    assert simulator.sensitivity_kernels is None
    assert simulator.receivers is _Parameters.receivers
