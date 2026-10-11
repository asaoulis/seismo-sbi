"""Building a variable-kernel CPS simulator leaves an existing Green's-function directory as it is."""
from seismo_sbi.simulators.cps.simulator import CPSVariableKernelSimulator
from seismo_sbi.simulators.receivers import Receivers


def test_construction_keeps_an_existing_storage_directory(tmp_path):
    (tmp_path / "earlier_run").mkdir()
    (tmp_path / "earlier_run" / "GF.mseed").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("kept")
    receivers = Receivers.from_arrays(["AAA"], ["XX"], [37.5], [-118.6])
    CPSVariableKernelSimulator(gf_storage_root=str(tmp_path), cps_path="/nonexistent", components="ZEN",
                               receivers=receivers, seismogram_duration_in_s=100,
                               synthetics_processing={"sampling_rate": 1.0, "filter": None})
    assert (tmp_path / "earlier_run" / "GF.mseed").read_bytes() == b"x"
    assert (tmp_path / "notes.txt").read_text() == "kept"
