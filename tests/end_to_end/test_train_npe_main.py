"""End-to-end run of the training launcher: a configuration file in, a trained checkpoint out.

Exercises the launcher itself — flag parsing, the typed training configuration, trainer
construction, the metric loggers and one training epoch — on the fabricated-kernel pipeline of
``test_train_npe_one_epoch``, so it needs no waveform database. The two pipeline steps that
require real simulations and noise on disk are supplied by the fixture instead.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismo_sbi.utils.seismograms import compute_data_vector_length
from seismo_sbi.sbi.training_data import TrainingData
from seismo_sbi.sbi.scalers import FlexibleScaler
from seismo_sbi.sbi.training_configuration import TrainingConfiguration

from tests.end_to_end.test_train_npe_one_epoch import (
    _build_kernel_pipeline, _DURATION, _SAMPLING_RATE, _NUM_SIMS,
)

pytestmark = pytest.mark.slow

LAUNCHER_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_NPE.py"
RUN_NAME = "launcher_smoke"
CONFIGURED_RUN_NAME = "npe_launcher"
JOB_NAME = "results"


def _load_launcher():
    """Import ``scripts/train_NPE.py``, which lives outside the package, by path."""
    spec = importlib.util.spec_from_file_location("train_npe_launcher", LAUNCHER_PATH)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def _write_configuration(tmp_path):
    """A complete, parseable configuration file; only its ``ml_*`` blocks reach the trainer."""
    (tmp_path / "stations.txt").write_text("PKD BK 35.945 -120.541\nORV BK 39.554 -121.500\n")
    (tmp_path / "components.json").write_text('{"PKD": ["Z"], "ORV": ["Z"]}')
    config_path = tmp_path / "npe_launcher.yaml"
    config_path.write_text(f"""
run_name: {CONFIGURED_RUN_NAME}
output_directory: '{tmp_path}'
job_name: {JOB_NAME}
generate_dataset: False
num_jobs: 1

ml_architecture: cnn
ml_batch:
  train: 8
  num_workers: 0
ml_optimizer:
  lr: 1.0e-4
  lr_schedule: constant
ml_logging:
  wandb: false
  csv: true

seismic_context:
  simulation_type: 'kernel'
  components: 'Z'
  stations_path: '{tmp_path / "stations.txt"}'
  station_components_path: '{tmp_path / "components.json"}'
  seismogram_duration: {_DURATION}
  sampling_rate: {_SAMPLING_RATE}
  syngine_address: null
  processing:
    filter_sampling_rate: 5.0
    sampling_rate: {_SAMPLING_RATE}

parameters:
  inference:
    moment_tensor:
      fiducial: [1.e+16, 1.e+16, 1.e+16, 1.e+16, 1.e+16, 1.e+16]
      stencil_deltas: [1.e+13, 1.e+13, 1.e+13, 1.e+13, 1.e+13, 1.e+13]
      bounds: [[-5.e+17, -5.e+17, -5.e+17, -5.e+17, -5.e+17, -5.e+17],
               [ 5.e+17,  5.e+17,  5.e+17,  5.e+17,  5.e+17,  5.e+17]]
  nuisance:
    source_location:
      fiducial: [35.0, -120.0, 10.0, 0.0]
      bounds: [[34.0, -121.0, 1.0, -2.0], [36.0, -119.0, 30.0, 2.0]]

simulations:
  num_simulations: {_NUM_SIMS}
  sampling_method:
    moment_tensor: "uniform"
    source_location: "constant"

compression:
  optimal_score:
    empirical_block: '{tmp_path}'

inference:
  sbi:
    method: 'posterior'
    pipeline: 'single_event'
    noise_model:
      type: 'gaussian'
      noise_level: 1.0e-06
  likelihood:
    run: False

jobs:
  real_events: {{}}
  simulations:
    random_events: 0
    fixed_events: []
    custom_events: {{}}
  noise_models: {{}}
  plots:
    disable_plotting: True
""")
    return config_path


@pytest.fixture(scope="module")
def launcher_run(tmp_path_factory):
    """Run ``main()`` once on the kernel pipeline and return where it wrote its outputs."""
    tmp_path = tmp_path_factory.mktemp("train_npe_main")
    pipeline, _, data_vector_length = _build_kernel_pipeline(tmp_path)
    # The kernel fixture writes its simulations into a 'train' subfolder.
    pipeline.simulations_output_path = pipeline.simulations_output_path + "/train"
    pipeline.training_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)

    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    simulation_paths = sorted(Path(pipeline.simulations_output_path).glob("*.h5"))
    data = TrainingData(
        simulation_paths=simulation_paths,
        components=pipeline.data_manager.data_loader.components,
        station_locations=pipeline.simulation_parameters.receivers.get_station_locations_array(),
        trace_length=trace_length,
        data_scaler=FlexibleScaler(pipeline.parameters),
        augmentation_chain=None, augmentation_nuisance_params={},
        post_noise_chain=None, post_noise_nuisance_params={},
    )

    config_path = _write_configuration(tmp_path)
    launcher = _load_launcher()
    patch = pytest.MonkeyPatch()
    patch.setattr(launcher, "build_pipeline", lambda *args, **kwargs: pipeline)
    patch.setattr(launcher, "generate_training_dataset", lambda *args, **kwargs: simulation_paths)
    patch.setattr(launcher, "prepare_training_data", lambda *args, **kwargs: data)
    patch.setattr(sys, "argv", ["train_NPE.py", "--config", str(config_path),
                                "--run-name", RUN_NAME, "--epochs", "1"])
    launcher.main()
    patch.undo()

    return pipeline.models_output_path / RUN_NAME


def test_the_launcher_writes_a_checkpoint(launcher_run):
    assert list((launcher_run / "checkpoints").glob("best_model-*.ckpt"))


def test_the_launcher_writes_the_csv_metrics_asked_for_in_the_config(launcher_run):
    metrics = (launcher_run / "metrics.csv").read_text()
    assert "val_loss" in metrics.splitlines()[0]


def test_the_sidecar_records_the_configured_architecture(launcher_run):
    import json
    meta = json.loads((launcher_run / "model_meta.json").read_text())
    assert meta["model_config"]["station_encoder"] == "cnn"
    assert meta["model_config"]["channels"] == TrainingConfiguration().model_dim
    assert meta["trace_length"] == compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    assert meta["model_config"]["theta_scaler"] == {"moment_tensor": "linear"}


def test_the_generate_stage_stops_before_training(tmp_path, monkeypatch):
    """``--stage generate`` returns once the simulations exist, without building a model."""
    launcher = _load_launcher()
    config_path = _write_configuration(tmp_path)
    monkeypatch.setattr(launcher, "build_pipeline", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(launcher, "generate_training_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr(launcher, "prepare_training_data", _fail_if_called)
    monkeypatch.setattr(sys, "argv", ["train_NPE.py", "--config", str(config_path),
                                      "--stage", "generate"])
    launcher.main()


def _fail_if_called(*args, **kwargs):
    raise AssertionError("the generate stage must not prepare training data")
