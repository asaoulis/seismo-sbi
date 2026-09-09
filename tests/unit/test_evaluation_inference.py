"""Selecting which real event an evaluation loads."""

import inspect
from types import SimpleNamespace

import pytest

from seismo_sbi.evaluation.inference import load_real_observation


def test_job_name_has_no_default():
    """The event key comes from the caller, so the library names no study's event."""
    assert (inspect.signature(load_real_observation).parameters["job_name"].default
            is inspect.Parameter.empty)


def test_an_unknown_job_name_lists_the_available_events():
    config = SimpleNamespace(real_event_jobs={"event1": "/events/event1.h5"})
    with pytest.raises(KeyError, match="event1"):
        load_real_observation(config, None, "not_an_event")
