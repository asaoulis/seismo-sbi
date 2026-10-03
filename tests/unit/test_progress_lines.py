"""The dataset-generation lines cluster job checks search for reach a launcher's stdout word for word."""
import logging
from types import SimpleNamespace

import pytest

from seismo_sbi.sbi.datasets.dataset_generator import DatasetGenerator
from seismo_sbi.sbi.datasets.training_data import generate_training_dataset
from seismo_sbi.utils.environment import log_progress_to_stdout


@pytest.fixture
def restore_library_logging():
    library_logger = logging.getLogger("seismo_sbi")
    handlers, level = list(library_logger.handlers), library_logger.level
    yield
    library_logger.handlers, library_logger.level = handlers, level


def test_the_skipped_simulations_line_reaches_stdout(restore_library_logging, capsys):
    log_progress_to_stdout()
    DatasetGenerator._guard_against_excessive_skips([True, False, True, True, True])

    assert capsys.readouterr().out == (
        "[dataset_generator] 1/5 simulations skipped (20.00%) after exhausting retries.\n")


def test_the_dataset_ready_line_reaches_stdout(restore_library_logging, capsys, tmp_path):
    log_progress_to_stdout()
    (tmp_path / "sim_0.h5").touch()
    pipeline = SimpleNamespace(simulations_output_path=str(tmp_path),
                               compute_data_vector_properties=lambda *args: None)
    config = SimpleNamespace(pipeline_parameters=SimpleNamespace(generate_dataset=False), real_event_jobs={})

    generate_training_dataset(pipeline, config, skip_compression_stencil=True)

    assert capsys.readouterr().out == f"Training dataset ready: 1 simulations at {tmp_path}\n"
