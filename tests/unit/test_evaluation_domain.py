"""Unit tests for the pluggable evaluation-domain loader
(``seismo_sbi.evaluation.domain``): ``EventSpec``, the ``EvalDomain`` protocol,
and ``load_domain`` (both the dotted ``module:Class`` form and the file-path
``/abs/adapter.py:Class`` form), plus the actionable errors on bad specs.
"""
import sys
import textwrap

import numpy as np
import pytest

from seismo_sbi.evaluation import EventSpec, EvalDomain, load_domain


_ADAPTER_SRC = textwrap.dedent(
    '''
    import numpy as np
    from seismo_sbi.evaluation import EventSpec

    class DummyDomain:
        name = "dummy"
        def discover_events(self, only):
            specs = [EventSpec(event="E1", job_name="E1", h5_path=None,
                               ref_mt=np.zeros(6), ref_lat=0.0, ref_lon=0.0,
                               ref_depth_km=1.0, extra={"flag": True})]
            return [s for s in specs if not only or s.event in only]
        def station_sets(self, spec, master_names):
            return list(master_names), list(master_names)
        def source_vec(self, spec, cond_param_map):
            return None if cond_param_map is None else np.zeros(1)
        def reference_overlay(self, spec, ml_ensembles, parameters, data_scaler, out_dir):
            return {}
        def event_summary(self, spec, ml_all, ml_filt):
            return {"ok": True}
    '''
)


def test_event_spec_fields_and_default_extra():
    s = EventSpec(event="No01", job_name="No01", h5_path="/x.h5",
                  ref_mt=np.zeros(6), ref_lat=36.4, ref_lon=25.5, ref_depth_km=12.0)
    assert s.event == "No01"
    assert s.extra == {}  # default_factory dict
    assert s.ref_depth_km == 12.0


def test_load_domain_file_path(tmp_path):
    f = tmp_path / "dummy_adapter.py"
    f.write_text(_ADAPTER_SRC)
    dom = load_domain(f"{f}:DummyDomain")
    assert dom.name == "dummy"
    # structural protocol satisfaction (EvalDomain is @runtime_checkable)
    assert isinstance(dom, EvalDomain)
    specs = dom.discover_events(None)
    assert len(specs) == 1 and specs[0].event == "E1"
    assert dom.discover_events({"nope"}) == []
    assert dom.event_summary(specs[0], np.zeros((2, 6)), np.zeros((2, 6))) == {"ok": True}


def test_load_domain_dotted_module(tmp_path, monkeypatch):
    pkgdir = tmp_path / "dotted_pkg_dir"
    pkgdir.mkdir()
    (pkgdir / "dotted_adapter.py").write_text(_ADAPTER_SRC)
    monkeypatch.syspath_prepend(str(pkgdir))
    sys.modules.pop("dotted_adapter", None)
    dom = load_domain("dotted_adapter:DummyDomain")
    assert dom.name == "dummy"


def test_load_domain_bad_spec_no_colon():
    with pytest.raises(ValueError):
        load_domain("no_colon_here")


def test_load_domain_missing_module():
    with pytest.raises(ImportError):
        load_domain("seismo_sbi.no_such_module_xyz:Whatever")


def test_load_domain_missing_class(tmp_path):
    f = tmp_path / "adapter_no_class.py"
    f.write_text(_ADAPTER_SRC)
    with pytest.raises(ImportError):
        load_domain(f"{f}:NotThere")


def test_load_domain_missing_file():
    with pytest.raises(ImportError):
        load_domain("/nonexistent/path/adapter.py:DummyDomain")
