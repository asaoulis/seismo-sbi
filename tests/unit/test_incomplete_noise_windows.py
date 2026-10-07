"""Incomplete real-noise windows + variable-station training.

`RealNoiseSampler` historically dropped a whole noise window if ANY model station was
missing, so usable windows = pool x P(all stations present). That fraction collapses as
the station count grows (about 78% of windows at 29 stations in one recorded pool),
which pushed the station set *down* exactly when more stations were wanted.

Under variable-station training a window missing station X is still perfectly good noise
for every station it does have. These tests pin that: absent stations are zero-filled and
reported via a mask, the station draw is restricted to that mask, and the default path is
byte-identical to the old behaviour.
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
    vec, present = loader.load_flattened_simulation_vector_with_presence(f)
    assert present.tolist() == [True, True, True]
    assert vec.size == len(NAMES) * 3 * NPTS


def test_missing_station_is_zero_filled_and_flagged(tmp_path, loader):
    f = tmp_path / "partial.h5"
    _write(f, ["AAA", "CCC"])                      # BBB absent
    vec, present = loader.load_flattened_simulation_vector_with_presence(f)
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
    vec, present = loader.load_flattened_simulation_vector_with_presence(f)
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

