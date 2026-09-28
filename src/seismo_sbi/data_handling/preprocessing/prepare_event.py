"""One recorded event, from raw MiniSEED and StationXML on disk to the SBI event file.

:class:`PreprocessingConfiguration` mirrors a ``preprocessing:`` YAML block; :func:`prepare_event`
reads each station's waveforms around the event, removes the instrument response, band-passes
and resamples them, keeps a MiniSEED copy of the processed stream, and writes the event window
with the pre-event noise autocorrelations as the HDF5 file the pipeline reads.
"""
import logging
from dataclasses import MISSING, dataclass, fields
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import obspy
import yaml
from obspy import UTCDateTime

from seismo_sbi.data_handling.preprocessing.io import find_mseed_files, load_inventory, load_waveforms, write_window
from seismo_sbi.data_handling.preprocessing.processing import deconvolve_and_filter
from seismo_sbi.data_handling.preprocessing.sbi_export import export_to_sbi_h5
from seismo_sbi.simulators.receivers import Receivers
from seismo_sbi.utils.errors import InvalidConfiguration

logger = logging.getLogger(__name__)

#: Data read on each side of the covariance and event windows, in s, so filtering has room to settle.
LOAD_PADDING_S = 120


@dataclass
class BandpassFilter:
    """The band-pass applied after response removal (``preprocessing.filter``)."""

    #: Low corner frequency in Hz.
    freqmin_hz: float
    #: High corner frequency in Hz.
    freqmax_hz: float
    corners: int = 4
    zerophase: bool = False


@dataclass
class PreprocessingConfiguration:
    """The ``preprocessing:`` block. Relative paths are taken from the working directory.

    ``event_start_utc`` and ``event_end_utc`` are ISO 8601 times; ``prefilter_hz`` gives the
    four corner frequencies of the response-removal taper (None keeps the library default);
    ``event_file`` names the HDF5 file written under ``<output_dir>/events``.
    """

    data_dir: str
    output_dir: str
    stations_file: str
    event_name: str
    event_start_utc: str
    event_end_utc: str
    sampling_rate_hz: float
    filter: BandpassFilter
    prefilter_hz: Optional[List[float]] = None
    stationxml_dir: Optional[str] = None
    covariance_window_s: float = 300.0
    remove_response: bool = True
    channel_glob: str = "BH?"
    event_file: Optional[str] = None

    @classmethod
    def from_yaml_block(cls, block):
        """The configuration from the parsed ``preprocessing`` mapping."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - known)
        missing = [f.name for f in fields(cls) if f.default is MISSING and f.name not in block]
        if unknown or missing:
            raise InvalidConfiguration(f"preprocessing block: unknown keys {unknown}, missing keys {missing}")
        return cls(**{**block, "filter": BandpassFilter(**block["filter"])})

    @classmethod
    def from_yaml(cls, path):
        """The configuration from the ``preprocessing:`` block of the YAML file at ``path``."""
        with open(path, "r", encoding="utf-8") as stream:
            return cls.from_yaml_block(yaml.safe_load(stream)["preprocessing"])


def prepare_event(config: PreprocessingConfiguration) -> Path:
    """Write the event file for ``config`` and return its path.

    A station is kept only if all three components cover the event window.
    """
    event_start = datetime.fromisoformat(config.event_start_utc)
    event_end = datetime.fromisoformat(config.event_end_utc)
    cov_window = timedelta(seconds=config.covariance_window_s)
    station_pairs = Receivers.from_station_file(config.stations_file).network_station_codes()
    logger.info(f"Loaded {len(station_pairs)} stations from {config.stations_file}")

    t0_load = UTCDateTime(event_start) - cov_window.total_seconds() - LOAD_PADDING_S
    t1_load = UTCDateTime(event_end) + LOAD_PADDING_S
    combined_stream, available_stations = load_station_waveforms(config, station_pairs, t0_load, t1_load)
    inventory, remove_response = load_response_inventory(config)

    logger.info("Processing waveforms...")
    processed_stream = deconvolve_and_filter(
        combined_stream,
        inventory=inventory,
        remove_response=remove_response,
        prefilter_kwargs=None if config.prefilter_hz is None else dict(pre_filt=list(config.prefilter_hz)),
        filter_kwargs=dict(freqmin=config.filter.freqmin_hz, freqmax=config.filter.freqmax_hz,
                           corners=config.filter.corners, zerophase=config.filter.zerophase),
        target_sr=config.sampling_rate_hz,
    )
    return write_event_files(config, processed_stream, available_stations, (event_start, event_end), cov_window)


def load_station_waveforms(config, station_pairs, t0_load, t1_load):
    """``(stream, station_names)`` of every station with data between ``t0_load`` and ``t1_load``."""
    combined_stream = obspy.Stream()
    available_stations = []

    for network, station in station_pairs:
        paths = find_mseed_files(
            Path(config.data_dir), station, t0_load, t1_load,
            network=network, channel_glob=config.channel_glob,
        )
        if not paths:
            logger.warning(f"  No data for {network}.{station} — skipping")
            continue
        try:
            st = load_waveforms(paths, starttime=t0_load, endtime=t1_load)
            if len(st) == 0:
                logger.warning(f"  Empty stream for {network}.{station} — skipping")
                continue
            combined_stream += st
            available_stations.append(station)
        except Exception as exc:
            logger.warning(f"  Could not load {network}.{station}: {exc}")
            continue

    logger.info(f"Available stations: {available_stations} ({len(available_stations)} total)")
    if len(available_stations) == 0:
        raise RuntimeError(f"No station in {config.stations_file} has data under {config.data_dir}.")
    return combined_stream, available_stations


def load_response_inventory(config):
    """``(inventory, remove_response)``: the StationXML inventory, or None and False when
    the response is not removed or no StationXML is found."""
    remove_response = config.remove_response
    stationxml_dir = Path(config.stationxml_dir or Path(config.data_dir) / "stationxml")
    inventory = None
    if remove_response:
        if not stationxml_dir.is_dir():
            logger.warning(
                f"stationxml_dir {stationxml_dir} not found — "
                "skipping response removal."
            )
            remove_response = False
        else:
            try:
                inventory = load_inventory(stationxml_dir)
                logger.info(f"Loaded inventory from {stationxml_dir}")
            except FileNotFoundError as exc:
                logger.warning(f"{exc} — skipping response removal.")
                remove_response = False
    return inventory, remove_response


def write_event_files(config, processed_stream, available_stations, event_window, cov_window):
    """Write the processed stream as MiniSEED and the event window as the SBI HDF5 file."""
    output_dir = Path(config.output_dir)
    daily_output_dir = output_dir / f"{config.event_name}_daily"
    daily_output_dir.mkdir(parents=True, exist_ok=True)
    daily_mseed = daily_output_dir / "preprocessed.mseed"
    write_window(processed_stream, daily_mseed)
    logger.info(f"Daily preprocessed mseed written to {daily_mseed}")

    event_output_dir = output_dir / "events"
    event_output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = event_output_dir / (config.event_file or f"{config.event_name}.h5")

    export_to_sbi_h5(
        stream=processed_stream,
        receivers=available_stations,
        event_window=event_window,
        out_path=h5_path,
        sampling_rate=config.sampling_rate_hz,
        covariance_window=cov_window,
        full_auto_correlation=True,
    )
    logger.info(f"Event h5 written to {h5_path}")
    return h5_path
