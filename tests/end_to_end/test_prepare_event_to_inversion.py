"""Raw MiniSEED of a known source, through prepare_event, recovers that source by least squares.

The raw records are Instaseis displacement at 5 Hz, and one station's horizontals are coded 1/2 at
azimuths 30/120. The event window opens SYNTHETICS_PRE_EVENT_PAD_S before the origin, so the
observed and synthetic time bases coincide.
"""
import os
from pathlib import Path

import h5py
import instaseis
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.core.inventory import Channel, Inventory, Network, Station
from pyrocko import moment_tensor as pmt
from pyrocko import orthodrome

from seismo_sbi.data_handling.preprocessing.prepare_event import PreprocessingConfiguration, prepare_event
from seismo_sbi.moment_tensor.comparison import from_pyrocko, kagan
from seismo_sbi.simulators.instaseis.querier import (SYNTHETICS_PRE_EVENT_PAD_S,
                                                     keep_inverse_mapping_out_of_the_numba_disk_cache)
from seismo_sbi.simulators.instaseis.simulator import InstaseisSourceSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.simulation_io import SimulationDataLoader, seismogram_map_to_array
from seismo_sbi.simulators.sources import build_stf_sliprate

INSTASEIS_DB = os.environ.get("INSTASEIS_DB", "")
ORIGIN = UTCDateTime("2020-03-01T12:00:00")
SOURCE = (37.0, -118.0, 15.0)
RAW_RATE_HZ = 5.0
DURATION_S = 200
AZIMUTHS_DEG = [15, 100, 190, 280]
ROTATED_STATION, AZIMUTH_1_DEG = "A100", 30.0

pytestmark = [pytest.mark.slow, pytest.mark.requires_data,
              pytest.mark.skipif(not Path(INSTASEIS_DB).exists(), reason="INSTASEIS_DB not set")]


def station_receivers():
    receivers = []
    for azimuth_deg in AZIMUTHS_DEG:
        latitude, longitude = orthodrome.azidist_to_latlon(SOURCE[0], SOURCE[1], azimuth_deg, 300 / 111.19)
        receivers.append(Receiver(float(latitude), float(longitude), "XX", f"A{azimuth_deg:03d}", ["Z", "E", "N"]))
    return Receivers(receivers=receivers)


def raw_records(receivers, m6):
    """``{station: {"Z", "N", "E": Trace}}`` at RAW_RATE_HZ from 30 minutes before the origin, for a
    Dirac moment-rate function as the simulator uses."""
    keep_inverse_mapping_out_of_the_numba_disk_cache()
    database = instaseis.open_db(INSTASEIS_DB)
    source = instaseis.Source(SOURCE[0], SOURCE[1], depth_in_m=SOURCE[2] * 1e3, origin_time=ORIGIN,
                              dt=database.info.dt, **dict(zip(["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"], m6)))
    source.set_sliprate(build_stf_sliprate(None, database.info.dt), database.info.dt, normalize=False)
    records = {}
    for receiver in receivers.iterate():
        stream = database.get_seismograms(source, instaseis.Receiver(receiver.latitude, receiver.longitude),
                                          "ZNE", kind="displacement", remove_source_shift=False, reconvolve_stf=True)
        stream.interpolate(RAW_RATE_HZ, method="lanczos", a=20)
        stream.trim(ORIGIN - 1800, ORIGIN + 1800, pad=True, fill_value=0.0)
        records[receiver.station_name] = {trace.stats.channel[-1]: trace for trace in stream}
    return records


def write_raw_day(data_dir, station, traces):
    day = data_dir / station / f"{ORIGIN.year}.{ORIGIN.julday:03d}"
    day.mkdir(parents=True)
    for channel, data in traces.items():
        trace = Trace(data, header={"network": "XX", "station": station, "channel": channel,
                                    "sampling_rate": RAW_RATE_HZ, "starttime": ORIGIN - 1800})
        Stream([trace]).write(str(day / f"XX.{station}..{channel}.{ORIGIN.year}.{ORIGIN.julday:03d}.mseed"),
                              format="MSEED", encoding="FLOAT64")


def write_raw_data(data_dir, receivers, records):
    stations = []
    for receiver in receivers.iterate():
        name = receiver.station_name
        vertical, north, east = (records[name][component].data for component in "ZNE")
        orientation = {"BHZ": (0.0, -90.0), "BHN": (0.0, 0.0), "BHE": (90.0, 0.0)}
        traces = {"BHZ": vertical, "BHN": north, "BHE": east}
        if name == ROTATED_STATION:
            azimuth = np.radians(AZIMUTH_1_DEG)
            traces = {"BHZ": vertical, "BH1": north * np.cos(azimuth) + east * np.sin(azimuth),
                      "BH2": -north * np.sin(azimuth) + east * np.cos(azimuth)}
            orientation = {"BHZ": (0.0, -90.0), "BH1": (AZIMUTH_1_DEG, 0.0), "BH2": (AZIMUTH_1_DEG + 90, 0.0)}
        write_raw_day(data_dir, name, traces)
        stations.append(Station(name, receiver.latitude, receiver.longitude, 0.0, channels=[
            Channel(code=code, location_code="", latitude=receiver.latitude, longitude=receiver.longitude,
                    elevation=0.0, depth=0.0, azimuth=azimuth_deg, dip=dip_deg, sample_rate=RAW_RATE_HZ)
            for code, (azimuth_deg, dip_deg) in orientation.items()]))
    (data_dir / "stationxml").mkdir()
    Inventory(networks=[Network("XX", stations=stations)], source="test").write(
        str(data_dir / "stationxml" / "XX.xml"), format="STATIONXML")


def test_prepare_event_then_least_squares_recovers_the_source(tmp_path):
    true_m6 = from_pyrocko(pmt.MomentTensor(strike=30, dip=60, rake=-80, scalar_moment=1e17))
    receivers = station_receivers()
    write_raw_data(tmp_path / "raw", receivers, raw_records(receivers, true_m6))
    (tmp_path / "stations.txt").write_text("".join(
        f"{receiver.station_name} XX {receiver.latitude} {receiver.longitude}\n" for receiver in receivers.iterate()))
    window_start = ORIGIN - SYNTHETICS_PRE_EVENT_PAD_S
    event_file = prepare_event(PreprocessingConfiguration.from_yaml_block({
        "data_dir": str(tmp_path / "raw"), "output_dir": str(tmp_path / "out"),
        "stations_file": str(tmp_path / "stations.txt"), "event_name": "known",
        "event_start_utc": window_start.isoformat(), "event_end_utc": (window_start + DURATION_S).isoformat(),
        "sampling_rate_hz": 1.0, "filter": {"freqmin_hz": 0.02, "freqmax_hz": 0.05},
        "remove_response": False, "covariance_window_s": 300}))

    simulator = InstaseisSourceSimulator(
        INSTASEIS_DB, components="ZEN", receivers=receivers, seismogram_duration_in_s=DURATION_S,
        synthetics_processing={"filter": {"type": "bandpass", "freqmin": 0.02, "freqmax": 0.05, "corners": 4,
                                          "zerophase": False}, "sampling_rate": 1.0,
                               "filter_sampling_rate": RAW_RATE_HZ})
    kernels = np.array([seismogram_map_to_array(simulator.run_simulation(
        {"source_location": [*SOURCE, 0.0], "moment_tensor": list(unit)})[1], receivers) for unit in np.eye(6)])
    with h5py.File(event_file, "r") as event:
        observed = SimulationDataLoader("ZEN", receivers).convert_sim_data_to_array(
            {"outputs": {station: {key: event["outputs"][station][key][()] for key in event["outputs"][station]}
                         for station in event["outputs"]}})
    recovered_m6 = np.linalg.solve(kernels @ kernels.T, kernels @ observed)

    predicted = kernels.T @ true_m6
    assert observed @ predicted / (np.linalg.norm(observed) * np.linalg.norm(predicted)) > 0.99
    assert kagan(recovered_m6, true_m6) < 1.0
    assert abs(np.linalg.norm(recovered_m6) / np.linalg.norm(true_m6) - 1) < 0.03
