"""Incomplete real-noise windows and variable-station training.

A window missing a model station is kept: the absent station is zero-filled and marked in a
presence mask, the station draw is restricted to that mask, a rescaled window is rescaled station by
station, and the default path still raises on a missing station.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from seismo_sbi.simulators.simulation_io import SimulationDataLoader
from seismo_sbi.sbi.npe.data.dataloading import StationSubsampler


class _Rec:
    def __init__(self, name, components):
        self.station_name = name
        self.components = components


class _Receivers:
    def __init__(self, names, components="ZNE"):
        self.receivers = [_Rec(n, components) for n in names]

    def iterate(self):
        return iter(self.receivers)


NAMES = ["AAA", "BBB", "CCC"]
NPTS = 8


def _write(path, present_names, components="ZNE"):
    with h5py.File(path, "w") as h:
        g = h.create_group("outputs")
        for i, n in enumerate(present_names):
            sg = g.create_group(n)
            for j, c in enumerate(components):
                sg.create_dataset(c, data=np.full(NPTS, float(i + 1) * (j + 1)))


@pytest.fixture
def loader():
    return SimulationDataLoader("ZNE", _Receivers(NAMES), NPTS)


def test_complete_window_reports_all_present(tmp_path, loader):
    f = tmp_path / "full.h5"
    _write(f, NAMES)
    vec, present = loader.load_simulation_data_array_with_presence(f)
    assert present.tolist() == [True, True, True]
    assert vec.size == len(NAMES) * 3 * NPTS


def test_missing_station_is_zero_filled_and_flagged(tmp_path, loader):
    f = tmp_path / "partial.h5"
    _write(f, ["AAA", "CCC"])                      # BBB absent
    vec, present = loader.load_simulation_data_array_with_presence(f)
    assert present.tolist() == [True, False, True]
    # canonical shape preserved, and the absent station's block is exactly zero
    assert vec.size == len(NAMES) * 3 * NPTS
    block = vec.reshape(len(NAMES), 3, NPTS)
    assert np.all(block[1] == 0.0)
    assert np.any(block[0] != 0.0) and np.any(block[2] != 0.0)


def test_station_missing_one_component_counts_as_absent(tmp_path, loader):
    """A part-zero station would otherwise reach the model as partly-real data."""
    f = tmp_path / "partcomp.h5"
    with h5py.File(f, "w") as h:
        g = h.create_group("outputs")
        for n in NAMES:
            sg = g.create_group(n)
            comps = "ZN" if n == "BBB" else "ZNE"   # BBB lacks E
            for j, c in enumerate(comps):
                sg.create_dataset(c, data=np.full(NPTS, 1.0 + j))
    vec, present = loader.load_simulation_data_array_with_presence(f)
    assert present.tolist() == [True, False, True]
    assert np.all(vec.reshape(len(NAMES), 3, NPTS)[1] == 0.0)


def test_default_path_still_raises_on_missing_station(tmp_path, loader):
    """The shared loader contract is unchanged: absence is an error unless opted out."""
    f = tmp_path / "partial2.h5"
    _write(f, ["AAA", "CCC"])
    with h5py.File(f, "r") as h:
        with pytest.raises(KeyError):
            loader.convert_sim_data_to_array(h)


def test_subsampler_unrestricted_by_default():
    sub = StationSubsampler(keep_fraction=1.0)
    keep = sub(5)
    assert keep.tolist() == [0, 1, 2, 3, 4]


def test_subsampler_never_keeps_an_unavailable_station():
    sub = StationSubsampler(keep_fraction=1.0)
    avail = np.array([True, False, True, False, True])
    for _ in range(25):
        keep = sub(5, available=avail)
        assert set(keep.tolist()) <= {0, 2, 4}
    # keep_fraction 1.0 over the available pool keeps exactly the available stations
    assert sub(5, available=avail).tolist() == [0, 2, 4]


def test_subsampler_fraction_applies_to_the_available_pool():
    sub = StationSubsampler(keep_fraction=0.5, min_stations=1)
    avail = np.array([True] * 4 + [False] * 6)
    for _ in range(25):
        keep = sub(10, available=avail)
        assert len(keep) == 2                      # 0.5 * 4 available, not 0.5 * 10
        assert set(keep.tolist()) <= {0, 1, 2, 3}


def test_subsampler_rejects_an_empty_availability_mask():
    sub = StationSubsampler(keep_fraction=1.0)
    with pytest.raises(ValueError):
        sub(3, available=np.zeros(3, dtype=bool))


def test_the_noise_model_passes_allow_incomplete_to_the_sampler(monkeypatch):
    """``allow_incomplete: true`` in a configuration reaches :class:`RealNoiseSampler`; without it the
    sampler would skip incomplete windows and quietly draw from a smaller pool."""
    import seismo_sbi.sbi.noises.noise_model as noise_model_mod

    seen = {}

    class _Spy:
        def __init__(self, *a, **kw):
            seen.update(kw)

    monkeypatch.setattr(noise_model_mod, "RealNoiseSampler", _Spy)
    for block, expected in (
        ({"type": "real_noise", "noise_level": 0.0, "noise_catalogue_path": "/x",
          "allow_incomplete": True, "rescale": False}, True),
        ({"type": "real_noise", "noise_level": 0.0, "noise_catalogue_path": "/x",
          "allow_incomplete": False}, False),
        ({"type": "real_noise", "noise_level": 0.0, "noise_catalogue_path": "/x"}, False),
    ):
        seen.clear()
        noise_model_mod.build_noise_sampler(noise_model_mod.NoiseModelConfiguration.from_yaml_block(block),
                                            object(), 100, 100)
        assert seen.get("allow_incomplete") is expected, block


def test_real_noise_training_needs_no_noise_level(monkeypatch):
    import seismo_sbi.sbi.noises.noise_model as noise_model_mod

    monkeypatch.setattr(noise_model_mod, "RealNoiseSampler", lambda *a, **kw: "sampler")
    noise_model = noise_model_mod.NoiseModelConfiguration("real_noise", noise_catalogue_path="/x")
    assert noise_model_mod.build_noise_sampler(noise_model, object(), 100, 100) == "sampler"
    assert noise_model_mod.build_test_noise_samplers([("real_noise", "/x")], noise_model, object(), 100, 100) == {
        "real_noise": "sampler"}



def _rescale_pool(directory, present_names, window_variance):
    """One noise window holding ``present_names``, each trace of variance-ratio interest."""
    directory.mkdir()
    with h5py.File(directory / "window.h5", "w") as h:
        outputs, misc = h.create_group("outputs"), h.create_group("misc")
        for station_index, name in enumerate(present_names):
            station_outputs, station_misc = outputs.create_group(name), misc.create_group(name)
            for component_index, component in enumerate("ZNE"):
                station_outputs.create_dataset(component, data=np.arange(NPTS) + 10.0 * station_index + component_index)
                station_misc.create_dataset(component, data=window_variance * np.exp(-np.arange(NPTS) / 3.0))
    return directory


def _rescale_sampler(directory):
    from seismo_sbi.sbi.noises.real_noise import RealNoiseSampler
    from seismo_sbi.simulators.receivers import Receiver, Receivers

    receivers = Receivers(receivers=[Receiver(0.0, float(k), "XX", name, ["Z", "N", "E"]) for k, name in enumerate(NAMES)])
    return RealNoiseSampler.from_receivers(receivers, "ZNE", NPTS, 1.0, directory, data_length=NPTS, allow_incomplete=True)


TARGET = {name: {component: np.array([2.0 + k]) for component in "ZNE"} for k, name in enumerate(NAMES)}


def test_an_incomplete_window_is_rescaled_station_by_station(tmp_path):
    sampler = _rescale_sampler(_rescale_pool(tmp_path / "pool", ["AAA", "CCC"], window_variance=8.0))
    sampler.rescale_to(TARGET)

    noise, present, covariance_data = sampler.draw()

    traces = noise.reshape(len(NAMES), 3, NPTS)
    assert present.tolist() == [True, False, True] and sorted(covariance_data) == ["AAA", "CCC"]
    assert np.allclose(traces[0, 1], (np.arange(NPTS) + 1.0) / np.sqrt(8.0 / 2.0))
    assert np.allclose(traces[2, 2], (np.arange(NPTS) + 12.0) / np.sqrt(8.0 / 4.0))
    assert not traces[1].any()


@pytest.mark.parametrize("present_names", [NAMES, ["AAA", "CCC"]])
def test_a_rescaled_draw_from_the_pool_equals_the_draw_from_disk(tmp_path, present_names):
    pool = _rescale_pool(tmp_path / "pool", present_names, window_variance=8.0)
    from_disk = _rescale_sampler(pool)
    from_disk.rescale_to(TARGET)
    pooled_after, pooled_before = _rescale_sampler(pool), _rescale_sampler(pool)
    pooled_after.preload_cache(max_workers=1, dtype=np.float64)
    pooled_after.rescale_to(TARGET)
    pooled_before.rescale_to(TARGET)
    pooled_before.preload_cache(max_workers=1, dtype=np.float64)

    expected = from_disk.draw()
    (pool / "window.h5").unlink()
    for pooled in (pooled_after, pooled_before):
        draw = pooled.draw()
        assert np.array_equal(draw.noise, expected.noise)
        assert np.array_equal(draw.present, expected.present)
