"""End-to-end smoke test for the ML neural-compression training flow.

Drives the full ``train_NPE`` path — dataset generation -> data loading with
synthetic noise -> embedding net + normalising flow -> one training epoch — on a
TINY synthetic dataset, headless (no W&B logging, no checkpoint files), on CPU.

Crucially it needs **no Instaseis/CPS database**: by fabricating a
``ScoreCompressionData`` (random sensitivity kernels) and forcing the
``FixedLocationKernelSimulator`` (pure NumPy ``kernels.T @ moment_tensor``), the
forward model is a matrix product. The pipeline auto-selects that simulator when
every non-``moment_tensor`` parameter uses ``constant`` sampling.

This is the **gate for new ML compression architectures**: register a new builder
in ``EMBEDDING_NET_REGISTRY`` (see ``sbi/compression/ML/train.py``) and add it to
``ARCHITECTURES`` below; the test will train it for one epoch and assert the
embedding-net -> flow path produces a finite log-probability.

Marked ``slow`` (heavier than unit tests; constructs a real transformer + flow),
but runs in well under a minute on CPU.
"""

import numpy as np
import pytest

from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer, EMBEDDING_NET_REGISTRY
from seismo_sbi.sbi.scalers import FlexibleScaler
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length
from seismo_sbi.sbi.types.parameters import (
    SimulationParameters,
    DatasetGenerationParameters,
    IterativeLeastSquaresParameters,
)

# Reuse the slow-test builders for receivers / model / pipeline.
from tests.end_to_end.test_pipeline_simulators import (
    _build_receivers,
    _build_mt_model_parameters,
    _build_pipeline,
)

pytestmark = pytest.mark.slow

# Architectures exercised by the one-epoch gate. New compression architectures
# should be added here once registered in EMBEDDING_NET_REGISTRY.
ARCHITECTURES = ["seismogram_transformer"]

# Station encoders exercised by the encoder-gate parametrized test (Phase 2).
# Each entry is (encoder_name, encoder_config) — tiny configs to keep CPU-fast.
STATION_ENCODERS = [
    ("cnn", {}),
    ("pno", {"width": 8, "modes": 4, "n_blocks": 2, "downsample": 4}),
    ("tcn", {"channels": 8, "n_blocks": 2, "kernel_size": 3, "downsample": 4}),
]

_DURATION = 200          # seconds; long enough that the CNN feature extractor does not
_SAMPLING_RATE = 1.0     # collapse the trace under its stride-2 down-sampling.
_NUM_SIMS = 24
_MODEL_DIM = 16          # tiny channels/latent dim to keep the test fast.


def _build_kernel_pipeline(tmp_path):
    """A SingleEventPipeline backed by the fabricated-kernel simulator (no Instaseis/CPS)."""
    receivers = _build_receivers()
    sim_params = SimulationParameters(
        receivers=receivers,
        components="Z",
        seismogram_duration=_DURATION,
        syngine_address=None,
        sampling_rate=_SAMPLING_RATE,
        processing={"sampling_rate": _SAMPLING_RATE},
        # Start in 'kernel' mode so the pipeline never opens an Instaseis DB. The kernel
        # simulator is built without data here and re-seeded with real (fabricated) kernels
        # by use_kernel_simulator_if_possible below.
        simulation_type="kernel",
    )
    model_params = _build_mt_model_parameters()
    dataset_params = DatasetGenerationParameters(
        num_simulations=_NUM_SIMS,
        sampling_method={"moment_tensor": "uniform", "source_location": "constant"},
        iterative_least_squares=IterativeLeastSquaresParameters(
            max_iterations=1, damping_factor=0.01,
        ),
    )

    pipeline = _build_pipeline(tmp_path, sim_params, model_params, dataset_params)

    # Per-trace length and total data-vector length (matches the pipeline's own arithmetic).
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    num_traces = sum(len(r.components) for r in receivers.iterate())
    data_vector_length = num_traces * trace_length

    # Fabricate sensitivity kernels: shape (6 MT components, total data-vector length).
    rng = np.random.default_rng(0)
    gradients = rng.standard_normal((6, data_vector_length)) * 1e-2
    score_data = ScoreCompressionData(
        theta_fiducial=np.asarray(model_params.theta_fiducial["moment_tensor"], dtype=float),
        data_fiducial=np.zeros(data_vector_length),
        data_parameter_gradients=gradients,
        second_order_gradients=None,
    )

    # Force the kernel simulator (all non-MT params are 'constant'), then generate data.
    pipeline.use_kernel_simulator_if_possible(score_data, dataset_params.sampling_method)
    pipeline.generate_simulation_data(dataset_params)

    return pipeline, dataset_params, data_vector_length


@pytest.fixture(scope="module")
def kernel_pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("train_npe_one_epoch")
    return _build_kernel_pipeline(tmp_path)


def _train_one_epoch(pipeline, data_vector_length, architecture, tmp_path):
    """Run a single headless training epoch and return the trained model."""
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)

    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)

    train_max_index = int(0.9 * _NUM_SIMS)
    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
        "random_shift_distribution": (0, 0),
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "train_shuffle": True,
        "val_shuffle": False,
        "num_workers": 0,
    }

    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM, architecture=architecture,
        trace_length=trace_length,
    )
    model = trainer.train(
        f"test_{architecture}", epochs=1, output_path=tmp_path,
        dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )
    return trainer, model, dataloader_args


def test_dataset_generated(kernel_pipeline):
    """The kernel simulator wrote a tiny training dataset to disk."""
    from pathlib import Path
    pipeline, _, _ = kernel_pipeline
    h5_files = list(Path(pipeline.simulations_output_path + "/train").glob("*.h5"))
    assert len(h5_files) >= _NUM_SIMS - 1, f"expected ~{_NUM_SIMS} sims, found {len(h5_files)}"


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_train_one_epoch_returns_finite_logprob(kernel_pipeline, tmp_path, architecture):
    """Each registered architecture trains for one epoch and yields a finite log-prob.

    This exercises the full embedding-net -> normalising-flow path end-to-end and is
    the gate any new compression architecture must pass.
    """
    import torch

    assert architecture in EMBEDDING_NET_REGISTRY, (
        f"{architecture} not registered in EMBEDDING_NET_REGISTRY"
    )

    pipeline, _, data_vector_length = kernel_pipeline
    trainer, model, dataloader_args = _train_one_epoch(
        pipeline, data_vector_length, architecture, tmp_path
    )

    assert model is not None
    from seismo_sbi.sbi.compression.ML.seismogram_transformer import NPELightningModule
    assert isinstance(model, NPELightningModule)

    # Pull one batch and check the flow's log-prob is finite.
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), "flow produced non-finite log-prob"


# Source-conditioning injection sets exercised by the conditioned one-epoch gate (Phase 3).
CONDITIONING_INJECTIONS = [
    ["token_add", "film"],
    ["relative_posemb", "concat_context"],
]


@pytest.mark.parametrize("inject", CONDITIONING_INJECTIONS, ids=["+".join(i) for i in CONDITIONING_INJECTIONS])
def test_conditioned_one_epoch(kernel_pipeline, tmp_path, inject):
    """Source-location-conditioned training runs one epoch and yields a finite log-prob.

    Source location is fed as RAW (lat, lon, depth) conditioning via the dataloader's
    conditioning_param_map; the embedding net unpacks it from the packed context. This is
    the Phase 3 gate; the unconditioned gates above guard backward compatibility.
    """
    import torch

    pipeline, _, data_vector_length = kernel_pipeline
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)

    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    train_max_index = int(0.9 * _NUM_SIMS)

    conditioning_param_map = {"source_location": ["latitude", "longitude", "depth"]}
    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
        "random_shift_distribution": (0, 0),
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "train_shuffle": True,
        "val_shuffle": False,
        "num_workers": 0,
        "conditioning_param_map": conditioning_param_map,
    }

    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer",
        trace_length=trace_length,
        model_config={"conditioning": {
            "n_cond": 3, "d_cond": 8, "coord_mode": "geographic", "inject": inject,
        }},
    )
    model = trainer.train(
        f"test_cond_{'_'.join(inject)}", epochs=1, output_path=tmp_path,
        dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )
    assert model is not None

    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    # Context is packed: (B, N*C*T + n_cond).
    assert x.dim() == 2
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), (
        f"conditioned model (inject={inject}) produced non-finite log-prob"
    )


def test_load_best_rebuilds_nondefault_architecture(kernel_pipeline, tmp_path):
    """Regression: train a non-default (PNO) checkpoint with checkpointing on, then reload it
    into a freshly DEFAULT-constructed trainer. load_best must rebuild the flow from the
    model_meta.json sidecar (architecture/model_config), not the loader's own default config,
    and produce a working model with a finite log-prob.
    """
    import torch
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders

    pipeline, _, data_vector_length = kernel_pipeline
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)
    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    train_max_index = int(0.9 * _NUM_SIMS)
    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
        "random_shift_distribution": (0, 0),
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
    }

    # Train a PNO model with checkpointing enabled so the .ckpt + model_meta.json are written.
    pno_cfg = {"station_encoder": "pno", "encoder_config": {"width": 8, "modes": 4, "n_blocks": 2, "downsample": 4}}
    trainer = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length, model_config=pno_cfg,
    )
    run_name = "pno_ckpt"
    trainer.train(run_name, epochs=1, output_path=tmp_path, dataloader_args=dataloader_args,
                  logger=None, enable_checkpointing=True, enable_progress_bar=False)

    # Fresh trainer built with the DEFAULT (cnn) config — structurally different from PNO.
    reloader = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
    )
    # Sanity: before reload it really is the cnn encoder, not pno.
    assert reloader.model.flow._embedding_net.station_encoder.__class__.__name__ == "CNNEncoder"

    reloader.load_best(tmp_path / run_name)   # must rebuild from the sidecar, not raise

    # After reload the embedding net must be the PNO encoder, and the model must run.
    assert reloader.model.flow._embedding_net.station_encoder.__class__.__name__ == "PhaseNeuralOperatorEncoder"
    reloader.model.to(reloader.device)
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    with torch.no_grad():
        log_prob = reloader.model(x.to(reloader.device), theta.to(reloader.device))
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("encoder_name,encoder_cfg", STATION_ENCODERS, ids=[e[0] for e in STATION_ENCODERS])
def test_station_encoder_one_epoch(kernel_pipeline, tmp_path, encoder_name, encoder_cfg):
    """Each registered station encoder trains for one epoch with the seismogram_transformer
    architecture and yields a finite log-probability.

    This is the gate for new per-station encoders (Phase 2+).  Keep encoder configs
    tiny so the test runs quickly on CPU.
    """
    import torch
    from seismo_sbi.sbi.compression.ML.station_encoders import PER_STATION_ENCODER_REGISTRY

    assert encoder_name in PER_STATION_ENCODER_REGISTRY, (
        f"Encoder '{encoder_name}' not registered in PER_STATION_ENCODER_REGISTRY"
    )

    pipeline, _, data_vector_length = kernel_pipeline
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)

    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    train_max_index = int(0.9 * _NUM_SIMS)

    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
        "random_shift_distribution": (0, 0),
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8,
        "val_batch_size": 8,
        "train_shuffle": True,
        "val_shuffle": False,
        "num_workers": 0,
    }

    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer",
        trace_length=trace_length,
        model_config={"station_encoder": encoder_name, "encoder_config": encoder_cfg},
    )
    model = trainer.train(
        f"test_encoder_{encoder_name}", epochs=1, output_path=tmp_path,
        dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )

    assert model is not None

    # Pull one validation batch and check finite log-prob.
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), (
        f"encoder '{encoder_name}' produced non-finite log-prob after one epoch"
    )
