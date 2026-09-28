"""The forward models read m6 as up-south-east: first-principles checks on Instaseis and CPS.

Each of the six elementary tensors is simulated at four stations 400 km from the source, band
passed to 33-100 s. Herrmann's expansion of the vertical surface-wave displacement gives
``Z(m_rp) / Z(m_rt) = -tan(az)`` and ``Z(m_tp) / (Z(m_tt) - Z(m_pp)) = -tan(2 az)`` whatever the
Earth model; an east-west mirror reverses both. The two backends must agree on the sign of every
elementary seismogram once the Instaseis pre-origin pad is allowed for. Needs ``INSTASEIS_DB`` and
``CPS_PATH``.
"""
import os
from pathlib import Path

import numpy as np
import pytest
from pyrocko import orthodrome

from seismo_sbi.simulators.cps.compatibility import load_velocity_model
from seismo_sbi.simulators.cps.simulator import CPSVariableKernelSimulator
from seismo_sbi.simulators.instaseis.querier import SYNTHETICS_PRE_EVENT_PAD_S
from seismo_sbi.simulators.instaseis.simulator import InstaseisSourceSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers

pytestmark = pytest.mark.slow

INSTASEIS_DB = os.environ.get("INSTASEIS_DB")
CPS_PATH = os.environ.get("CPS_PATH")
#: Layered model (thickness, vp, vs, density in km, km/s, g/cm^3, then Qp, Qs) close to PREM's crust.
PREM_LIKE_MODEL = Path(__file__).resolve().parents[1] / "fixtures" / "prem_like_cps_model.txt"
SOURCE_LOCATION = [37.0, -118.0, 15.0, 0.0]
STATION_AZIMUTHS_DEG = [20.0, 110.0, 200.0, 290.0]
EPICENTRAL_DISTANCE_KM = 400.0
PROCESSING = {"filter": {"type": "bandpass", "freqmin": 0.01, "freqmax": 0.03, "corners": 4,
                         "zerophase": False},
              "sampling_rate": 1.0, "filter_sampling_rate": 5.0}
COMPONENT_NAMES = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]


def receivers():
    stations = []
    distance_deg = np.degrees(EPICENTRAL_DISTANCE_KM * 1e3 / orthodrome.earthradius)
    for azimuth_deg in STATION_AZIMUTHS_DEG:
        lat, lon = orthodrome.azidist_to_latlon(*SOURCE_LOCATION[:2], azimuth_deg, distance_deg)
        stations.append(Receiver(float(lat), float(lon), "XX", f"A{int(azimuth_deg):03d}", ["Z", "E", "N"]))
    return Receivers(receivers=stations)


@pytest.fixture(scope="module")
def elementary_seismograms(tmp_path_factory):
    """``{backend: {m6 component name: {station: {component: trace}}}}`` for unit tensors of 1e17 N.m."""
    if not (INSTASEIS_DB and Path(INSTASEIS_DB).is_dir() and CPS_PATH and Path(CPS_PATH, "hprep96").exists()):
        pytest.skip("needs INSTASEIS_DB and CPS_PATH")
    common = dict(components="ZEN", receivers=receivers(), seismogram_duration_in_s=400,
                  synthetics_processing=PROCESSING)
    instaseis = InstaseisSourceSimulator(INSTASEIS_DB, **common)
    cps = CPSVariableKernelSimulator(gf_storage_root=str(tmp_path_factory.mktemp("cps")), cps_path=CPS_PATH,
                                     **common)
    velocity_model = load_velocity_model(str(PREM_LIKE_MODEL))
    seismograms = {"instaseis": {}, "cps": {}}
    for index, name in enumerate(COMPONENT_NAMES):
        source = {"source_location": SOURCE_LOCATION, "moment_tensor": list(1e17 * np.eye(6)[index])}
        seismograms["instaseis"][name] = instaseis.run_simulation(source)[1]
        seismograms["cps"][name] = cps.run_simulation({**source, "velocity_model": velocity_model})[1]
    return seismograms


def ratio_and_correlation(trace, reference):
    """Least-squares amplitude of ``trace`` against ``reference`` and their correlation coefficient."""
    ratio = np.dot(trace, reference) / np.dot(reference, reference)
    return ratio, np.dot(trace, reference) / np.linalg.norm(trace) / np.linalg.norm(reference)


def lagged_correlation(trace, reference, lag):
    """Correlation of ``trace`` with ``reference`` delayed by ``lag`` samples, over their common length."""
    n_samples = min(len(trace), len(reference))
    trace, reference = trace[:n_samples], reference[:n_samples]
    shifted = np.zeros_like(reference)
    shifted[lag:] = reference[:len(reference) - lag]
    return np.dot(trace, shifted) / np.linalg.norm(trace) / np.linalg.norm(shifted)


@pytest.mark.parametrize("backend", ["instaseis", "cps"])
def test_vertical_radiation_follows_herrmanns_azimuthal_pattern(elementary_seismograms, backend):
    seismograms = elementary_seismograms[backend]
    for azimuth_deg in STATION_AZIMUTHS_DEG:
        station = f"A{int(azimuth_deg):03d}"
        Z = {name: seismograms[name][station]["Z"] for name in COMPONENT_NAMES}
        dip_slip, dip_slip_cc = ratio_and_correlation(Z["m_rp"], Z["m_rt"])
        strike_slip, strike_slip_cc = ratio_and_correlation(Z["m_tp"], Z["m_tt"] - Z["m_pp"])
        np.testing.assert_allclose(dip_slip, -np.tan(np.radians(azimuth_deg)), rtol=0.03)
        np.testing.assert_allclose(strike_slip, -np.tan(2.0 * np.radians(azimuth_deg)), rtol=0.03)
        assert min(abs(dip_slip_cc), abs(strike_slip_cc)) > 0.99


@pytest.mark.parametrize("backend", ["instaseis", "cps"])
def test_an_m_rr_source_moves_no_station_transversely(elementary_seismograms, backend):
    for azimuth_deg in STATION_AZIMUTHS_DEG:
        traces = elementary_seismograms[backend]["m_rr"][f"A{int(azimuth_deg):03d}"]
        azimuth = np.radians(azimuth_deg)
        radial = traces["N"] * np.cos(azimuth) + traces["E"] * np.sin(azimuth)
        transverse = -traces["N"] * np.sin(azimuth) + traces["E"] * np.cos(azimuth)
        assert np.std(transverse) < 0.1 * np.std(radial)


def test_both_backends_give_every_elementary_seismogram_the_same_sign(elementary_seismograms):
    pad_samples = int(SYNTHETICS_PRE_EVENT_PAD_S * PROCESSING["sampling_rate"])
    largest = max(np.abs(trace).max() for station in elementary_seismograms["instaseis"].values()
                  for traces in station.values() for trace in traces.values())
    for name in COMPONENT_NAMES:
        for station, traces in elementary_seismograms["instaseis"][name].items():
            for component, instaseis_trace in traces.items():
                if np.abs(instaseis_trace).max() < 0.05 * largest:
                    continue
                cps_trace = elementary_seismograms["cps"][name][station][component]
                correlations = [lagged_correlation(instaseis_trace, cps_trace, pad_samples + lag)
                                for lag in range(-12, 13)]
                assert max(correlations, key=abs) > 0.8, (name, station, component)


def test_vertical_amplitudes_agree_between_backends(elementary_seismograms):
    for name in COMPONENT_NAMES:
        for station in elementary_seismograms["instaseis"][name]:
            instaseis_peak = np.abs(elementary_seismograms["instaseis"][name][station]["Z"]).max()
            cps_peak = np.abs(elementary_seismograms["cps"][name][station]["Z"]).max()
            station_peak = max(np.abs(elementary_seismograms["instaseis"][other][station]["Z"]).max()
                               for other in COMPONENT_NAMES)
            if instaseis_peak > 0.05 * station_peak:
                assert 0.7 < cps_peak / instaseis_peak < 1.4, (name, station)


def test_instaseis_synthetics_lead_cps_by_the_pre_origin_pad(elementary_seismograms):
    instaseis_trace = elementary_seismograms["instaseis"]["m_rr"]["A020"]["Z"]
    cps_trace = elementary_seismograms["cps"]["m_rr"]["A020"]["Z"]
    lags = range(0, 100)
    best_lag = max(lags, key=lambda lag: lagged_correlation(instaseis_trace, cps_trace, lag))
    assert abs(best_lag - SYNTHETICS_PRE_EVENT_PAD_S * PROCESSING["sampling_rate"]) <= 12
