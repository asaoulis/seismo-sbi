"""Where a source time sits on the moment-rate function: the triangle's peak by default, its onset for
CPS and for networks trained before the convention existed."""
import os

import numpy as np
import pytest

from seismo_sbi.sbi.npe.training.train import recorded_stf_alignment
from seismo_sbi.simulators.cps.simulator import CPSSimulator
from seismo_sbi.simulators.instaseis.querier import InstaseisDBQuerier
from seismo_sbi.simulators.receivers import Receiver, Receivers
from seismo_sbi.simulators.sources import (GeneralMomentTensor, GenericPointSource, SourceLocation,
                                           build_stf_sliprate, sliprate_centroid_s)
from seismo_sbi.utils.errors import InvalidConfiguration

DT_S = 4.87
MOMENT_TENSOR = np.array([0.0, -5e18, 5e18, 1e18, 0.0, 0.0])
RECEIVERS = Receivers(receivers=[Receiver(10.0, 0.0, "XX", "AAA", ["Z", "E", "N"])])


def _querier(stf_alignment):
    """A querier with only the attributes the source construction reads (no database opened)."""
    querier = object.__new__(InstaseisDBQuerier)
    querier._dt, querier.source_depth_offset_km, querier.stf_alignment = DT_S, 0.0, stf_alignment
    return querier


def _source(time_shift_s):
    return GenericPointSource(SourceLocation(35.7, -117.5, 12.0, time_shift_s), GeneralMomentTensor(MOMENT_TENSOR))


class NoKernelCPSSimulator(CPSSimulator):

    def compute_or_load_greens_functions(self, objstats, velocity_model, **kwargs):
        raise NotImplementedError


def test_the_dirac_sliprate_has_its_centroid_at_its_first_sample():
    assert sliprate_centroid_s(build_stf_sliprate(None, DT_S), DT_S) == 0.0


def test_a_symmetric_sliprate_has_its_centroid_at_its_peak():
    sliprate = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    assert sliprate_centroid_s(sliprate, 0.5) == pytest.approx(1.0)


def test_peak_alignment_puts_the_triangle_centroid_at_the_source_time():
    source = _querier("peak")._create_source_object(_source(9.4), stf_duration=2.0)
    centroid_s = source.time_shift + sliprate_centroid_s(source.sliprate, DT_S)
    assert centroid_s == pytest.approx(9.4, abs=1e-9)
    assert sliprate_centroid_s(source.sliprate, DT_S) > DT_S


def test_onset_alignment_starts_the_triangle_at_the_source_time():
    assert _querier("onset")._create_source_object(_source(9.4), stf_duration=2.0).time_shift == 9.4


def test_a_dirac_sits_at_the_source_time_under_either_alignment():
    for alignment in ("peak", "onset"):
        assert _querier(alignment)._create_source_object(_source(9.4)).time_shift == 9.4


def test_cps_reports_onset_and_rejects_peak():
    assert NoKernelCPSSimulator(None, None, "ZEN", RECEIVERS, 200, {"sampling_rate": 1}).stf_alignment == "onset"
    with pytest.raises(InvalidConfiguration, match="stf_alignment"):
        NoKernelCPSSimulator(None, None, "ZEN", RECEIVERS, 200, {"sampling_rate": 1}, stf_alignment="peak")


def test_a_sidecar_without_the_convention_is_onset_and_a_new_one_is_read_back():
    assert recorded_stf_alignment({"model_config": {"theta_scaler": {}}}) == "onset"
    assert recorded_stf_alignment({"model_config": {"stf_alignment": "peak"}}) == "peak"


DATABASE_20S = os.environ.get("INSTASEIS_DB_20S", "/data/shared/prem_a_20s")


@pytest.mark.requires_data
@pytest.mark.skipif(not os.path.isdir(DATABASE_20S), reason="needs an Instaseis database resolving 20 s")
def test_dirac_and_triangle_at_the_same_source_time_peak_within_half_a_sample():
    from seismo_sbi.simulators.instaseis.simulator import InstaseisSourceSimulator
    from seismo_sbi.simulators.simulation_io import seismogram_map_to_array

    receivers = Receivers(receivers=[Receiver(36.5, -115.16, "NN", "SHP", ["Z", "E", "N"]),
                                     Receiver(38.44, -120.72, "BK", "WELL", ["Z", "E", "N"])])
    processing = {"filter": {"type": "bandpass", "freqmin": 0.02, "freqmax": 0.05, "corners": 4,
                             "zerophase": False}, "sampling_rate": 1.0, "filter_sampling_rate": 1.0}
    lags_s = {}
    for alignment in ("peak", "onset"):
        simulator = InstaseisSourceSimulator(DATABASE_20S, components="ZEN", receivers=receivers,
                                             seismogram_duration_in_s=350.0, synthetics_processing=processing,
                                             stf_alignment=alignment)
        run = lambda **extra: seismogram_map_to_array(simulator.run_simulation(
            {"source_location": [35.7, -117.5, 12.0, 9.4], "moment_tensor": list(MOMENT_TENSOR), **extra})[1], receivers)
        dirac, triangle = run(), run(stf_duration=1.0)
        correlation = np.correlate(triangle, dirac, mode="full")
        k = int(np.argmax(correlation))
        vertex = 0.5 * (correlation[k - 1] - correlation[k + 1]) / (correlation[k - 1] - 2 * correlation[k] + correlation[k + 1])
        lags_s[alignment] = (k + vertex - (len(dirac) - 1)) / processing["sampling_rate"]
    assert abs(lags_s["peak"]) < 0.5 / processing["sampling_rate"]
    assert lags_s["onset"] > 0.5 * DT_S
