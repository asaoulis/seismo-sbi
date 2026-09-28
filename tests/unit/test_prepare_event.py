"""The preprocessing configuration block and the event file prepare_event writes from raw MiniSEED."""
from pathlib import Path

import h5py
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from seismo_sbi.data_handling.preprocessing.prepare_event import PreprocessingConfiguration, prepare_event
from seismo_sbi.utils.errors import InvalidConfiguration

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
START = UTCDateTime("2020-03-01T12:00:00")


def test_the_lv2_preprocessing_block_parses():
    config = PreprocessingConfiguration.from_yaml(EXAMPLES / "configs" / "LV2_preprocessing.yaml")

    assert (config.event_name, config.sampling_rate_hz, config.filter.freqmin_hz) == ("LV2", 1.0, 0.02)
    assert config.prefilter_hz == [0.005, 0.01, 0.2, 0.4]


def test_a_block_without_a_sampling_rate_is_rejected():
    block = {"data_dir": "d", "output_dir": "o", "stations_file": "s", "event_name": "e",
             "event_start_utc": "2020-03-01T12:00:00", "event_end_utc": "2020-03-01T12:01:00",
             "filter": {"freqmin_hz": 0.02, "freqmax_hz": 0.05}}

    with pytest.raises(InvalidConfiguration, match="sampling_rate_hz"):
        PreprocessingConfiguration.from_yaml_block(block)


def test_an_unknown_filter_key_is_rejected():
    block = {"data_dir": "d", "output_dir": "o", "stations_file": "s", "event_name": "e",
             "event_start_utc": "2020-03-01T12:00:00", "event_end_utc": "2020-03-01T12:01:00",
             "sampling_rate_hz": 1.0, "filter": {"freqmin_hz": 0.02, "freqmax_hz": 0.05, "type": "bandpass"}}

    with pytest.raises(InvalidConfiguration, match="filter.type"):
        PreprocessingConfiguration.from_yaml_block(block)


def _write_raw_day(data_dir, station, rng):
    day = data_dir / station / f"{START.year}.{START.julday:03d}"
    day.mkdir(parents=True)
    for channel in ("BHZ", "BHE", "BHN"):
        trace = Trace(rng.normal(size=4 * 3600 * 20), header=dict(
            network="XX", station=station, channel=channel, sampling_rate=20.0, starttime=START - 7200))
        Stream([trace]).write(str(day / f"XX.{station}..{channel}.{START.year}.{START.julday:03d}.mseed"),
                              format="MSEED")


def test_prepare_event_writes_every_three_component_station(tmp_path):
    rng = np.random.default_rng(0)
    for station in ("AAA", "BBB"):
        _write_raw_day(tmp_path / "raw", station, rng)
    (tmp_path / "stations.txt").write_text("AAA XX 37.0 -118.0\nBBB XX 38.0 -119.0\nCCC XX 39.0 -120.0\n")
    config = PreprocessingConfiguration.from_yaml_block({
        "data_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "out"),
        "stations_file": str(tmp_path / "stations.txt"), "event_name": "test",
        "event_start_utc": "2020-03-01T12:00:00", "event_end_utc": "2020-03-01T12:03:20",
        "sampling_rate_hz": 1.0, "filter": {"freqmin_hz": 0.02, "freqmax_hz": 0.05},
        "remove_response": False, "covariance_window_s": 600})

    event_file = prepare_event(config)

    with h5py.File(event_file, "r") as event:
        assert sorted(event["outputs"]) == ["AAA", "BBB"]
        assert {event["outputs"]["AAA"][component].shape for component in ("Z", "1", "2")} == {(201,)}
        assert event["misc"]["BBB"]["Z"].shape == (601,)
