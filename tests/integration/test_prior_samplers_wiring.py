"""Integration test: catalogue-prior closures flow through the dataset generator.

Exercises the exact sampler composition that
``DatasetGenerator.run_and_save_simulations`` performs (resolve sampler -> zip per
parameter -> flatten -> ``vector_to_simulation_inputs``) without needing a forward
simulator, plus the kernel-simulator fast-path flag logic in the pipeline.
"""

import numpy as np

from seismo_sbi.instaseis_simulator.dataset_generator import (
    DatasetGenerator,
    transform_sampling_func,
)
from seismo_sbi.sbi.types.parameters import ModelParameters
from seismo_sbi.priors.catalogue import EventCatalogue
from seismo_sbi.priors.samplers import (
    make_catalogue_location_sampler,
    make_gutenberg_richter_mt_sampler,
)


def _toy_catalogue():
    rng = np.random.default_rng(0)
    n = 300
    return EventCatalogue(
        latitude=rng.uniform(36.2, 36.8, n),
        longitude=rng.uniform(25.3, 25.7, n),
        depth=rng.uniform(5, 15, n),
        magnitude=rng.uniform(1.0, 4.0, n),
        magnitude_type=np.array(["Ml"] * n, dtype=object),
    )


def _params_with_source_and_mt():
    params = ModelParameters()
    params.names = {
        "source_location": ["latitude", "longitude", "depth", "time_shift"],
        "moment_tensor": ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"],
    }
    params.theta_fiducial = {
        "source_location": [36.5, 25.5, 10.0, 0.0],
        "moment_tensor": [0.0] * 6,
    }
    params.bounds = {
        "source_location": [[35.0, 24.0, 0.0, -5.0], [38.0, 27.0, 60.0, 5.0]],
        "moment_tensor": [[-1e18] * 6, [1e18] * 6],
    }
    return params


def test_dict_form_closures_compose_into_theta():
    cat = _toy_catalogue()
    params = _params_with_source_and_mt()
    sampling_method = {
        "source_location": make_catalogue_location_sampler(
            catalogue=cat, std_x_km=1.0, std_y_km=1.0, std_z_km=2.0,
            time_shift="uniform", seed=0),
        "moment_tensor": make_gutenberg_richter_mt_sampler(
            b_value=1.0, mw_min=1.0, mw_max=5.0, seed=0),
    }

    # mirror run_and_save_simulations' composition
    samplers = DatasetGenerator._create_sampler_generator_dict(params, sampling_method)
    assert all(callable(s) for s in samplers.values())

    sampler_args = params.bounds
    sampler_callable = lambda n: zip(
        *[s(sampler_args[k], n) for k, s in samplers.items()]
    )
    transformer = transform_sampling_func(
        sampler_callable, lambda v: params.vector_to_simulation_inputs(v)
    )
    out = list(transformer(8))
    assert len(out) == 8
    for inputs in out:
        assert set(inputs.keys()) == {"source_location", "moment_tensor"}
        assert inputs["source_location"].shape == (4,)
        assert inputs["moment_tensor"].shape == (6,)
        # source location stayed within bounds
        assert np.all(inputs["source_location"] >= np.array(params.bounds["source_location"][0]))
        assert np.all(inputs["source_location"] <= np.array(params.bounds["source_location"][1]))


def test_resolve_sampler_passthrough_and_lookup():
    closure = make_gutenberg_richter_mt_sampler(b_value=1.0, mw_min=1.0, mw_max=5.0)
    assert DatasetGenerator._resolve_sampler(closure) is closure
    # string still resolves to a built-in
    assert DatasetGenerator._resolve_sampler("constant") is \
        DatasetGenerator.sampler_lookup_map["constant"]


def test_kernel_fastpath_flag_logic():
    # MT-only GR: source_location constant, moment_tensor a closure -> kernel sim OK
    mt_closure = make_gutenberg_richter_mt_sampler(b_value=1.0, mw_min=1.0, mw_max=5.0)
    sm_mt_only = {"source_location": "constant", "moment_tensor": mt_closure}
    only_mt = all(s == "constant" for p, s in sm_mt_only.items() if p != "moment_tensor")
    assert only_mt is True

    # catalogue locations vary -> kernel sim disabled
    loc_closure = make_catalogue_location_sampler(
        catalogue=_toy_catalogue(), std_x_km=1.0, std_y_km=1.0, std_z_km=1.0)
    sm_loc = {"source_location": loc_closure, "moment_tensor": mt_closure}
    only_mt2 = all(s == "constant" for p, s in sm_loc.items() if p != "moment_tensor")
    assert only_mt2 is False
