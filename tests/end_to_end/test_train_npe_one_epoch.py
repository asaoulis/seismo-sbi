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


def _build_kernel_pipeline(tmp_path, receivers=None, components="Z", sampling_method=None):
    """A SingleEventPipeline backed by the fabricated-kernel simulator (no Instaseis/CPS).

    ``receivers`` / ``components`` default to the shared single-Z setup; pass a
    multi-component receivers + components string to exercise per-channel behaviour
    (e.g. component dropout). ``sampling_method`` overrides the per-parameter sampler
    selection (e.g. to route ``moment_tensor`` through a catalogue-prior closure);
    it must keep every non-``moment_tensor`` parameter ``constant`` so the kernel
    simulator is still selected.
    """
    receivers = receivers if receivers is not None else _build_receivers()
    sim_params = SimulationParameters(
        receivers=receivers,
        components=components,
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
    if sampling_method is None:
        sampling_method = {"moment_tensor": "uniform", "source_location": "constant"}
    dataset_params = DatasetGenerationParameters(
        num_simulations=_NUM_SIMS,
        sampling_method=sampling_method,
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


@pytest.fixture(scope="module")
def multicomp_kernel_pipeline(tmp_path_factory):
    """Two stations, two components (Z,N) — lets component dropout actually zero channels."""
    from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
    receivers = Receivers(receivers=[
        Receiver(35.945, -120.541, "BK", "PKD", ["Z", "N"]),
        Receiver(39.554, -121.500, "BK", "ORV", ["Z", "N"]),
    ])
    tmp_path = tmp_path_factory.mktemp("train_npe_one_epoch_multicomp")
    return _build_kernel_pipeline(tmp_path, receivers=receivers, components="ZN")


def _train_one_epoch(pipeline, data_vector_length, architecture, tmp_path, data_scaler=None,
                     model_config=None, flow_config=None, lr_second_stage="cosine",
                     enable_checkpointing=False, run_name=None):
    """Run a single headless training epoch and return the trained model."""
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    if data_scaler is None:
        data_scaler = FlexibleScaler(pipeline.parameters)

    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)

    train_max_index = int(0.9 * _NUM_SIMS)
    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
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
        model_config=model_config,
        flow_config=flow_config,
        lr_second_stage=lr_second_stage,
    )
    model = trainer.train(
        run_name or f"test_{architecture}", epochs=1, output_path=tmp_path,
        dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=enable_checkpointing, enable_progress_bar=False,
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


def test_flow_config_and_constant_lr_one_epoch_and_round_trip(kernel_pipeline, tmp_path):
    """The NDE-head/LR follow-up knobs train end-to-end and round-trip.

    Exercises a deeper flow (num_transforms=8) with the constant-after-warmup LR schedule for
    one headless epoch (checkpointing on), confirms a finite log-prob, and reloads from the
    model_meta.json sidecar — asserting the rebuilt flow keeps the 8 coupling transforms.
    """
    import torch
    from pyknos.nflows import transforms
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders

    pipeline, _, data_vector_length = kernel_pipeline

    def _count_coupling(flow):
        return sum(isinstance(t, transforms.PiecewiseRationalQuadraticCouplingTransform)
                   for t in flow._transform._transforms)

    run_name = "bigflow_constlr"
    trainer, model, dataloader_args = _train_one_epoch(
        pipeline, data_vector_length, "seismogram_transformer", tmp_path,
        flow_config={"num_transforms": 8}, lr_second_stage="constant",
        enable_checkpointing=True, run_name=run_name,
    )
    assert _count_coupling(trainer.flow) == 8
    assert model.lr_second_stage == "constant"

    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all()

    # Reload into a fresh DEFAULT trainer (num_transforms=5) — must rebuild to 8 from sidecar.
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    reloader = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
    )
    assert _count_coupling(reloader.flow) == 5            # before reload (default)
    reloader.load_best(tmp_path / run_name)               # rebuilds from model_meta.json
    assert _count_coupling(reloader.flow) == 8            # after reload (from sidecar)
    reloader.model.to(reloader.device)
    with torch.no_grad():
        log_prob2 = reloader.model(x.to(reloader.device), theta.to(reloader.device))
    assert torch.isfinite(log_prob2).all()


# Per-station amplitude embedding modes exercised by the one-epoch gate.
AMPLITUDE_MODES = ["array_relative", "absolute"]


@pytest.mark.parametrize("mode", AMPLITUDE_MODES)
def test_amplitude_embedding_one_epoch(kernel_pipeline, tmp_path, mode):
    """One headless epoch with the per-station amplitude embedding yields a finite log-prob,
    exercising the full encoder → amplitude-token → transformer → flow path end-to-end."""
    import torch

    pipeline, _, data_vector_length = kernel_pipeline
    model_config = {"amplitude_embedding": {"mode": mode}}
    trainer, model, dataloader_args = _train_one_epoch(
        pipeline, data_vector_length, "seismogram_transformer", tmp_path,
        model_config=model_config,
    )

    assert model is not None
    emb_net = model.flow._embedding_net
    assert getattr(emb_net, "amplitude_embedding", None) is not None, (
        "amplitude embedding was not wired into the trained model"
    )

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


def test_amplitude_distance_snr_one_epoch(kernel_pipeline, tmp_path):
    """Distance-corrected, SNR-weighted amplitude embedding under source-location conditioning
    trains one epoch to a finite log-prob (full conditioned + amplitude path end-to-end)."""
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
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
        "conditioning_param_map": conditioning_param_map,
    }

    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
        model_config={
            "conditioning": {"n_cond": 3, "d_cond": 8, "coord_mode": "geographic",
                             "inject": ["relative_posemb"]},
            "amplitude_embedding": {"mode": "array_relative",
                                    "distance_correction": True, "snr_weighting": True},
        },
    )
    model = trainer.train(
        "test_amp_dist_snr", epochs=1, output_path=tmp_path, dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )
    assert model is not None
    assert model.flow._embedding_net.amplitude_embedding.uses_distance is True

    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all()


@pytest.mark.parametrize("coords_mode", ["absolute", "relative"])
def test_variable_stations_one_epoch(kernel_pipeline, tmp_path, coords_mode):
    """Variable-station training (random subsampling + ragged pad/mask batching) runs one
    epoch and yields a finite log-prob, for both absolute and source-relative coords.

    Exercises the StationSubsampler, the variable_station_collate (ragged batches → padded
    context + mask), the fully-padded-station NaN guard, and per-sample coordinate encoding.
    Relative mode additionally requires source-location conditioning so a source vector is
    packed alongside the seismograms.
    """
    import torch
    from seismo_sbi.sbi.compression.ML.dataloading import (
        make_torch_dataloaders, StationSubsampler,
    )

    pipeline, _, data_vector_length = kernel_pipeline
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)
    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    train_max_index = int(0.9 * _NUM_SIMS)

    # keep_fraction 0.5–1.0 over 2 stations ⇒ N ∈ {1, 2} ⇒ ragged batches + fully-padded rows.
    subsampler = StationSubsampler(keep_fraction=(0.5, 1.0), min_stations=1)
    model_config = {"variable_stations": True, "station_coords_mode": coords_mode}
    conditioning_param_map = None
    if coords_mode == "relative":
        conditioning_param_map = {"source_location": ["latitude", "longitude", "depth"]}
        model_config["conditioning"] = {
            "n_cond": 3, "d_cond": 8, "coord_mode": "geographic", "inject": [],
        }

    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
        "conditioning_param_map": conditioning_param_map,
        "station_subsampler": subsampler,
    }

    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
        model_config=model_config,
    )
    model = trainer.train(
        f"test_varstations_{coords_mode}", epochs=1, output_path=tmp_path,
        dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )
    assert model is not None

    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    assert x.dim() == 2  # packed variable-station context
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), (
        f"variable-station model (coords_mode={coords_mode}) produced non-finite log-prob"
    )


def test_component_dropout_one_epoch(multicomp_kernel_pipeline, tmp_path):
    """A one-epoch run with a post-noise component_dropout chain stays finite while channels
    are actually being zeroed (2-component data, keep >=1/station). Exercises the full real
    training path: build post-noise chain -> dataloader applies it after noise -> flow."""
    import torch
    from seismo_sbi.instaseis_simulator.post_processing import (
        PostProcessingChain, ComponentDropoutEffect,
    )
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders

    pipeline, _, data_vector_length = multicomp_kernel_pipeline
    components = pipeline.data_manager.data_loader.components
    station_locations = pipeline.simulation_parameters.receivers.get_station_locations_array()
    data_scaler = FlexibleScaler(pipeline.parameters)
    synthetic_noise_sampler = lambda: np.random.normal(0.0, 1.0, data_vector_length)
    trace_length = compute_data_vector_length(_DURATION, _SAMPLING_RATE) + 1
    train_max_index = int(0.9 * _NUM_SIMS)

    # Aggressive drop probability so channels are frequently zeroed during the epoch.
    post_chain = PostProcessingChain([ComponentDropoutEffect()])
    assert post_chain.effects, "post-noise chain should be non-empty"

    dataloader_args = {
        "data_loader": pipeline.data_manager.data_loader,
        "data_folder": pipeline.simulations_output_path + "/train",
        "parameter_name_map": pipeline.parameters.names,
        "synthetic_noise_model_sampler": synthetic_noise_sampler,
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
        "post_noise_augmentation_chain": post_chain,
        "post_noise_nuisance_params": {"component_dropout": 0.5},
    }

    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
    )
    model = trainer.train(
        "test_component_dropout", epochs=1, output_path=tmp_path,
        dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )
    assert model is not None

    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), "component-dropout training produced non-finite log-prob"


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


def test_load_best_restores_amplitude_embedding(kernel_pipeline, tmp_path):
    """The amplitude_embedding config must round-trip through model_meta.json: train with it,
    reload into a default-constructed trainer (which has no amplitude embedding), and confirm
    it is restored and runs. Locks the inference-time persistence claim."""
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
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
    }

    amp_cfg = {"amplitude_embedding": {"mode": "array_relative", "num_freqs": 8, "sigma": 1.0}}
    trainer = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length, model_config=amp_cfg,
    )
    run_name = "amp_ckpt"
    trainer.train(run_name, epochs=1, output_path=tmp_path, dataloader_args=dataloader_args,
                  logger=None, enable_checkpointing=True, enable_progress_bar=False)

    reloader = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
    )
    assert reloader.model.flow._embedding_net.amplitude_embedding is None   # default has none

    reloader.load_best(tmp_path / run_name)

    restored = reloader.model.flow._embedding_net.amplitude_embedding
    assert restored is not None and restored.mode == "array_relative"
    reloader.model.to(reloader.device)
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    with torch.no_grad():
        log_prob = reloader.model(x.to(reloader.device), theta.to(reloader.device))
    assert torch.isfinite(log_prob).all()


def test_positional_encoding_one_epoch(kernel_pipeline, tmp_path):
    """One headless epoch with the §3.2 RFF station positional encoding (source-relative
    geometry + depth, injected every layer) under source-location conditioning yields a finite
    log-prob — the full encoder → RFF-posenc-conditioned transformer → flow path end-to-end."""
    import torch
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders

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
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
        "conditioning_param_map": conditioning_param_map,
    }

    trainer = CompressionTrainer(
        components, station_locations,
        channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
        model_config={
            "conditioning": {"n_cond": 3, "d_cond": 8, "coord_mode": "geographic",
                             "inject": ["relative_posemb"]},
            "positional_encoding": {"mode": "fourier", "include_depth": True,
                                    "inject_every_layer": True, "num_freqs": 8},
        },
    )
    model = trainer.train(
        "test_posenc_fourier", epochs=1, output_path=tmp_path, dataloader_args=dataloader_args,
        logger=None, enable_checkpointing=False, enable_progress_bar=False,
    )
    assert model is not None
    sp = model.flow._embedding_net.all_station_transformer.station_posenc
    assert sp is not None and sp.coords_kind == "relative" and sp.include_depth

    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), "RFF-posenc training produced non-finite log-prob"


def test_load_best_restores_positional_encoding(kernel_pipeline, tmp_path):
    """The positional_encoding config must round-trip through model_meta.json: train with it,
    reload into a default-constructed trainer (which has no posenc), and confirm it is restored
    and runs. Uses the absolute (unconditioned) variant so no conditioning plumbing is needed."""
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
        "data_scaler": data_scaler,
        "train_max_index": train_max_index,
        "train_batch_size": 8, "val_batch_size": 8,
        "train_shuffle": True, "val_shuffle": False, "num_workers": 0,
    }

    pe_cfg = {"positional_encoding": {"mode": "fourier", "num_freqs": 8, "sigma": 1.0}}
    trainer = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length, model_config=pe_cfg,
    )
    run_name = "posenc_ckpt"
    trainer.train(run_name, epochs=1, output_path=tmp_path, dataloader_args=dataloader_args,
                  logger=None, enable_checkpointing=True, enable_progress_bar=False)

    reloader = CompressionTrainer(
        components, station_locations, channels=_MODEL_DIM, latent_dim=_MODEL_DIM,
        architecture="seismogram_transformer", trace_length=trace_length,
    )
    assert reloader.model.flow._embedding_net.all_station_transformer.station_posenc is None

    reloader.load_best(tmp_path / run_name)

    restored = reloader.model.flow._embedding_net.all_station_transformer.station_posenc
    assert restored is not None and restored.coords_kind == "absolute"
    reloader.model.to(reloader.device)
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    with torch.no_grad():
        log_prob = reloader.model(x.to(reloader.device), theta.to(reloader.device))
    assert torch.isfinite(log_prob).all()


def test_gutenberg_richter_mt_prior_one_epoch(tmp_path):
    """A Gutenberg-Richter moment-tensor prior closure flows through dataset generation,
    keeps the kernel simulator engaged (source_location stays 'constant'), and trains one
    epoch to a finite log-prob.

    The magnitude range (Mw 2.0–3.5) is chosen so the truncated-GR M0 stays inside the
    model's ±5e14 component bounds, so FlexibleScaler maps the sampled tensors into [0,1].
    This is the dataset-generation gate for the catalogue moment-tensor prior.
    """
    import torch
    from seismo_sbi.priors.samplers import make_gutenberg_richter_mt_sampler
    from seismo_sbi.sbi.compression.ML.dataloading import make_torch_dataloaders

    gr_closure = make_gutenberg_richter_mt_sampler(
        b_value=1.0, mw_min=2.0, mw_max=3.5, seed=0,
    )
    sampling_method = {"moment_tensor": gr_closure, "source_location": "constant"}

    pipeline, _, data_vector_length = _build_kernel_pipeline(
        tmp_path, sampling_method=sampling_method,
    )
    # kernel simulator must still be the active forward model (locations are constant)
    from seismo_sbi.instaseis_simulator.simulator import FixedLocationKernelSimulator
    assert isinstance(pipeline.simulator_wrapper.simulator, FixedLocationKernelSimulator)

    from pathlib import Path
    h5_files = list(Path(pipeline.simulations_output_path + "/train").glob("*.h5"))
    assert len(h5_files) >= _NUM_SIMS - 1

    # Train through the scale_shape MomentTensorScaler (the parametrisation motivated by
    # the GR prior's many-orders-of-magnitude M0 range), and confirm it round-trips.
    mt_scaler = FlexibleScaler(pipeline.parameters, moment_tensor_scaling="scale_shape")
    full = np.array(
        pipeline.parameters.parameter_to_vector("theta_fiducial"), dtype=float
    ).reshape(1, -1)
    assert np.allclose(mt_scaler.inverse_transform(mt_scaler.transform(full)), full, rtol=1e-6)

    trainer, model, dataloader_args = _train_one_epoch(
        pipeline, data_vector_length, "seismogram_transformer", tmp_path,
        data_scaler=mt_scaler,
    )
    assert model is not None
    _, val_loader = make_torch_dataloaders(**dataloader_args)
    theta, x = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        log_prob = model(x.to(model.device), theta.to(model.device))
    assert torch.isfinite(log_prob).all(), "GR-MT-prior training produced non-finite log-prob"


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
