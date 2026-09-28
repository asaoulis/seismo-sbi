"""Launcher environment helpers: settings that must be in place before the science imports."""

import datetime

import pytest

from seismo_sbi.utils.environment import stamp_arviz_daily_warning

platformdirs = pytest.importorskip("platformdirs")


def test_arviz_stamp_holds_today_and_leaves_no_temp_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    stamp_arviz_daily_warning()
    stamp_arviz_daily_warning()
    stamp_dir = tmp_path / "arviz"
    assert (stamp_dir / "daily_warning").read_text() == datetime.date.today().isoformat()
    assert sorted(path.name for path in stamp_dir.iterdir()) == ["daily_warning"]


def test_arviz_stamp_replaces_a_stale_date(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    stamp_dir = tmp_path / "arviz"
    stamp_dir.mkdir()
    (stamp_dir / "daily_warning").write_text("2000-01-01")
    stamp_arviz_daily_warning()
    assert (stamp_dir / "daily_warning").read_text() == datetime.date.today().isoformat()


def test_library_progress_records_reach_stdout_as_bare_messages(capsys):
    import logging

    from seismo_sbi.utils.environment import log_progress_to_stdout

    library_logger = logging.getLogger("seismo_sbi")
    handlers, level = list(library_logger.handlers), library_logger.level
    try:
        log_progress_to_stdout()
        log_progress_to_stdout()
        logging.getLogger("seismo_sbi.sbi.pipeline").info("Starting MLE")
        logging.getLogger("seismo_sbi.sbi.pipeline").debug("not shown")
    finally:
        library_logger.handlers, library_logger.level = handlers, level

    assert capsys.readouterr().out == "Starting MLE\n"
