"""CPS seismograms follow Herrmann's azimuthal radiation pattern for the up-south-east m6.

The ten CPS Green's functions are replaced by random waveforms, so only the rotation to each
station azimuth and the moment-tensor contraction are tested:
``Z(m_rp) / Z(m_rt) = -tan(az)`` and ``Z(m_tp) / (Z(m_tt) - Z(m_pp)) = -tan(2 az)``.
"""
import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth

from seismo_sbi.simulators.cps.CPS import update_with_Gtensor
from seismo_sbi.simulators.cps.simulator import CPSSimulator
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.sources import GeneralMomentTensor, GenericPointSource, SourceLocation

N_SAMPLES = 64
#: Receiver azimuths from the source, in degrees, one per quadrant.
STATION_AZIMUTHS_DEG = [20.0, 110.0, 200.0, 290.0]
SOURCE = SourceLocation(0.0, 0.0, 10.0, 0.0)
#: The CPS Green's-function order: ZDD RDD ZDS RDS TDS ZSS RSS TSS ZEX REX.
N_GREEN_FUNCTIONS = 10


class RandomGreenFunctionCPSSimulator(CPSSimulator):
    """Reads one set of random Green's functions, written to ``gf_storage_root``, at every distance."""

    def compute_or_load_greens_functions(self, objstats, velocity_model, **kwargs):
        distances_km = np.unique(np.round(sorted([stats.distance for stats in objstats]), 1))
        waveforms = np.random.default_rng(1).standard_normal((N_GREEN_FUNCTIONS, N_SAMPLES + 8))
        stream = Stream()
        for distance_index in range(len(distances_km)):
            for waveform in waveforms:
                header = {"station": "%03d" % (distance_index + 1), "delta": 1.0, "starttime": UTCDateTime(0)}
                stream.append(Trace(waveform, header=header))
        stream.write(str(self.gf_storage_root / "GF.mseed"), format="MSEED")
        return update_with_Gtensor(objstats, None, delta=None, verbose=False,
                                   gf_directory=self.gf_storage_root)


def station_at(azimuth_deg):
    azimuth = np.radians(azimuth_deg)
    return Receiver(3.0 * np.cos(azimuth), 3.0 * np.sin(azimuth), "XX", f"A{int(azimuth_deg):03d}",
                    ["Z", "E", "N"])


def vertical_seismograms(tmp_path):
    """Station name to ``{m6 component name: vertical seismogram}`` for each unit component."""
    receivers = Receivers(receivers=[station_at(azimuth) for azimuth in STATION_AZIMUTHS_DEG])
    simulator = RandomGreenFunctionCPSSimulator(tmp_path, None, "ZEN", receivers, N_SAMPLES,
                                                {"sampling_rate": 1.0})
    names = ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]
    seismograms = {receiver.station_name: {} for receiver in receivers.iterate()}
    for index, name in enumerate(names):
        source = GenericPointSource(SOURCE, GeneralMomentTensor(list(1e15 * np.eye(6)[index])))
        for station, traces in simulator.generic_point_source_simulation(source).items():
            seismograms[station][name] = traces["Z"]
    return receivers, seismograms


def amplitude_ratio(trace, reference):
    """Least-squares amplitude of ``trace`` against ``reference``."""
    return np.dot(trace, reference) / np.dot(reference, reference)


@pytest.mark.parametrize("pattern", ["dip_slip", "strike_slip"])
def test_cps_vertical_radiation_follows_the_station_azimuth(tmp_path, pattern):
    receivers, seismograms = vertical_seismograms(tmp_path)
    for receiver in receivers.iterate():
        _, azimuth_deg, _ = gps2dist_azimuth(0.0, 0.0, receiver.latitude, receiver.longitude)
        Z = seismograms[receiver.station_name]
        if pattern == "dip_slip":
            ratio, predicted = amplitude_ratio(Z["m_rp"], Z["m_rt"]), -np.tan(np.radians(azimuth_deg))
        else:
            ratio = amplitude_ratio(Z["m_tp"], Z["m_tt"] - Z["m_pp"])
            predicted = -np.tan(2.0 * np.radians(azimuth_deg))
        np.testing.assert_allclose(ratio, predicted, rtol=1e-6)
