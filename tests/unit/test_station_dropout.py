"""
Unit tests for seismo_sbi.sbi.compression.ML.station_dropout — station-config
selection / fractional dropout and the shared packed-subset sample loop. Pure
numpy + a stub posterior; no trained model or pyrocko needed.
"""
import numpy as np
import pytest

from seismo_sbi.sbi.compression.ML.station_dropout import (
    StationConfig, make_dropout_configs, config_from_kept,
    sample_station_dropout_ensemble,
)

NAMES = [f"ST{i:02d}" for i in range(10)]


def test_full_first_and_subset_sizes():
    cfgs = make_dropout_configs(NAMES, keep_fraction=0.6, n_subsets=4,
                                min_stations=3, seed=0)
    assert len(cfgs) == 5
    full = cfgs[0]
    assert full.n == 10 and full.dropped == [] and list(full.keep) == list(range(10))
    for c in cfgs[1:]:
        assert c.n == 6                                   # round(0.6 * 10)
        assert len(c.dropped) == 4
        assert sorted(c.kept + c.dropped) == sorted(NAMES)
        assert c.kept == [NAMES[i] for i in c.keep]       # index<->name alignment


def test_subsets_are_distinct():
    cfgs = make_dropout_configs(NAMES, keep_fraction=0.6, n_subsets=4, seed=0)
    keys = [frozenset(c.keep.tolist()) for c in cfgs[1:]]
    assert len(set(keys)) == len(keys)


def test_reproducible_and_seed_sensitive():
    a = make_dropout_configs(NAMES, seed=7)
    b = make_dropout_configs(NAMES, seed=7)
    assert [list(c.keep) for c in a] == [list(c.keep) for c in b]
    c = make_dropout_configs(NAMES, seed=8)
    assert [list(x.keep) for x in a] != [list(x.keep) for x in c]


def test_min_stations_floor():
    cfgs = make_dropout_configs(NAMES, keep_fraction=0.1, n_subsets=3,
                                min_stations=3, seed=0)
    for c in cfgs[1:]:
        assert c.n == 3                                   # floored up from round(0.1*10)=1


def test_include_full_false():
    cfgs = make_dropout_configs(NAMES, n_subsets=2, include_full=False, seed=0)
    assert len(cfgs) == 2 and all(c.dropped for c in cfgs)


def test_infeasible_n_subsets_warns(capsys):
    # only 2 distinct size-1 subsets of {A, B}; requesting 5 must warn and cap.
    cfgs = make_dropout_configs(["A", "B"], keep_fraction=0.5, n_subsets=5,
                                min_stations=1, seed=0)
    assert "WARNING" in capsys.readouterr().out
    assert len(cfgs) - 1 <= 2


def test_station_config_n_and_as_dict():
    c = StationConfig(label="x", keep=np.array([0, 2, 4]),
                      kept=["a", "c", "e"], dropped=["b", "d"])
    assert c.n == 3
    d = c.as_dict()
    assert d["n"] == 3 and d["keep_indices"] == [0, 2, 4]
    assert d["kept"] == ["a", "c", "e"] and d["dropped"] == ["b", "d"]


def test_config_from_kept():
    c = config_from_kept(NAMES, ["ST00", "ST03", "ST09"], "filtered")
    assert list(c.keep) == [0, 3, 9]
    assert c.kept == ["ST00", "ST03", "ST09"]
    assert "ST01" in c.dropped and len(c.dropped) == 7


def test_empty_names_raises():
    with pytest.raises(ValueError):
        make_dropout_configs([], n_subsets=1)


def test_config_from_kept_drives_sample_loop():
    """Mirror the Santorini all-vs-filtered comparison: one full observation, two
    config_from_kept configs (all + a filtered subset ⊆ all), sampled together. The
    filtered config must pack a NARROWER context (fewer station rows) — proving the
    subset rows/coords are indexed consistently."""
    torch = pytest.importorskip("torch")
    names = [f"S{i}" for i in range(6)]
    filt = ["S0", "S2", "S4"]
    obs = np.random.default_rng(0).normal(size=(6, 3, 4))
    coords = np.random.default_rng(1).normal(size=(6, 2))
    cfg_all = config_from_kept(names, names, "ML all")
    cfg_filt = config_from_kept(names, filt, "ML filtered")
    assert cfg_filt.n == 3 and list(cfg_filt.keep) == [0, 2, 4]

    widths = []

    class StubPosterior:
        def sample(self, shape, ctx, show_progress_bars=False):
            widths.append(int(ctx.shape[-1]))
            return torch.zeros((shape[0], 6))

    class IdScaler:
        def inverse_transform(self, x):
            return np.asarray(x)

    ens, results = sample_station_dropout_ensemble(
        StubPosterior(), obs, coords, [cfg_all, cfg_filt], IdScaler(),
        num_samples=3, device="cpu")
    assert list(ens.keys()) == ["ML all", "ML filtered"]
    assert ens["ML all"].samples.shape == (3, 6)
    assert ens["ML filtered"].samples.shape == (3, 6)
    assert widths[0] > widths[1]                         # all packs more rows than filtered


def test_sample_station_dropout_ensemble_with_stub():
    torch = pytest.importorskip("torch")
    cfgs = make_dropout_configs(NAMES, n_subsets=2, seed=0)
    N, C, T = 10, 3, 5
    obs = np.random.default_rng(0).normal(size=(N, C, T))
    coords = np.random.default_rng(1).normal(size=(N, 2))

    class StubPosterior:
        def __init__(self):
            self.seen_widths = []

        def sample(self, shape, ctx, show_progress_bars=False):
            # context width must shrink with the kept-station count (packed subset path)
            self.seen_widths.append(int(ctx.shape[-1]))
            return torch.zeros((shape[0], 6))

    class IdScaler:
        def inverse_transform(self, x):
            return np.asarray(x)

    post = StubPosterior()
    ensemble, results = sample_station_dropout_ensemble(
        post, obs, coords, cfgs, IdScaler(),
        num_samples=4, device="cpu", event_name="ev")

    assert list(ensemble.keys()) == [c.label for c in cfgs]
    assert len(results) == len(cfgs)
    for c in cfgs:
        assert ensemble[c.label].samples.shape == (4, 6)
        assert ensemble[c.label].theta0 is None
    assert results[0].event_name.startswith("ev:")
    assert results[0].inversion_config.inversion_method == "ml_compressor"
    # full config packs a wider context than the dropped subsets
    assert post.seen_widths[0] > post.seen_widths[1]
