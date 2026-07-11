"""Shared helpers for the dataloader/augmentation benchmark scripts.

NOT a test and NOT a production entry point — support code for
``scripts/bench_aug_dataloader.py`` (per-sample attribution) and
``scripts/bench_training_workers.py`` (real-training worker sweep), so the
fabricated-kernel pipeline build and the augmentation config live in ONE place.

Reuses the slow-test pipeline builders (no Instaseis/CPS, no network).
"""
import numpy as np

from seismo_sbi.instaseis_simulator.receivers import Receiver, Receivers
from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length
from seismo_sbi.instaseis_simulator.post_processing import build_augmentation_chain
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
from seismo_sbi.sbi.types.parameters import (
    SimulationParameters,
    DatasetGenerationParameters,
    IterativeLeastSquaresParameters,
)
from tests.end_to_end.test_pipeline_simulators import (
    _build_mt_model_parameters,
    _build_pipeline,
)


def spread_receivers(n, components):
    """N stations on a lat/lon grid around the SoCal source, all with `components`."""
    comps = list(components)
    recs = [
        Receiver(34.0 + 0.7 * (i % 7), -122.0 + 0.6 * (i // 7), "BK", f"S{i:03d}", comps)
        for i in range(n)
    ]
    return Receivers(receivers=recs)


def build_kernel_pipeline(stations, components, num_sims, duration, sampling_rate, tmp_dir):
    """Build a kernel-simulator SingleEventPipeline and generate `num_sims` sims to disk.

    Returns ``(pipeline, data_vector_length, trace_length)``. Sims land under
    ``pipeline.simulations_output_path + '/train'``.
    """
    receivers = spread_receivers(stations, components)
    sim_params = SimulationParameters(
        receivers=receivers, components=components, seismogram_duration=duration,
        syngine_address=None, sampling_rate=sampling_rate,
        processing={"sampling_rate": sampling_rate}, simulation_type="kernel",
    )
    model_params = _build_mt_model_parameters()
    dataset_params = DatasetGenerationParameters(
        num_simulations=num_sims,
        sampling_method={"moment_tensor": "uniform", "source_location": "constant"},
        iterative_least_squares=IterativeLeastSquaresParameters(
            max_iterations=1, damping_factor=0.01,
        ),
    )
    pipeline = _build_pipeline(tmp_dir, sim_params, model_params, dataset_params)

    trace_length = compute_data_vector_length(duration, sampling_rate) + 1
    num_traces = sum(len(r.components) for r in receivers.iterate())
    data_vector_length = num_traces * trace_length

    rng = np.random.default_rng(0)
    score_data = ScoreCompressionData(
        theta_fiducial=np.asarray(model_params.theta_fiducial["moment_tensor"], dtype=float),
        data_fiducial=np.zeros(data_vector_length),
        data_parameter_gradients=rng.standard_normal((6, data_vector_length)) * 1e-2,
        second_order_gradients=None,
    )
    pipeline.use_kernel_simulator_if_possible(score_data, dataset_params.sampling_method)
    pipeline.generate_simulation_data(dataset_params)
    return pipeline, data_vector_length, trace_length


def production_model_config(model_dim=256, station_encoder="tcn", downsample=None):
    """The model_config + flow_config that the brustle-lomax production YAML resolves to.

    Mirrors ``scripts/train_NPE.py``'s config→model_config translation for
    ``scripts/configs/santorini/first_ml_npe_brustle_lomax.yaml`` so the benches exercise
    the REAL production architecture (tcn encoder + source-location conditioning +
    variable stations + amplitude embedding + RFF posenc + PMA-tokens pooling + an
    8-transform NSF flow). ``n_cond=3`` = (latitude, longitude, depth).

    Returns ``(model_config, flow_config, n_cond)``.
    """
    n_cond = 3
    model_config = {
        "station_encoder": station_encoder,
        "conditioning": {
            "n_cond": n_cond,
            "d_cond": model_dim,
            "coord_mode": "geographic",
            "inject": ["token_add", "film"],
            "n_fourier": 0,
        },
        "variable_stations": True,
        "station_coords_mode": "relative",
        "amplitude_embedding": {
            "mode": "array_relative",
            "per_component": False,
            "reference": "mean",
            "num_freqs": 16,
            "sigma": 1.0,
            "learnable_freqs": False,
            "scale": 1.0,
            "distance_correction": True,
            "snr_weighting": True,
            "snr_floor_quantile": 0.2,
        },
        "positional_encoding": {
            "mode": "fourier",
            "num_freqs": 16,
            "sigma": 1.0,
            "learnable_freqs": False,
            "include_depth": True,
            "inject_every_layer": True,
            "standardize": "running",
        },
        "pma_pooling": {
            "pool_over": "tokens",
            "num_seeds": 4,
            "combine": "linear",
            "ffn": True,
            "seed_self_attention": False,
        },
    }
    if downsample is not None:
        model_config["encoder_config"] = {"downsample": int(downsample)}
    flow_config = {"num_transforms": 8}
    return model_config, flow_config, n_cond


def make_variable_station_batch(batch_size, n_master, components, trace_length, n_cond,
                                keep_fraction=(0.5, 1.0), min_stations=3, seed=0,
                                source_fiducial=(36.45, 25.55, 12.0),
                                station_center=(36.45, 25.55), station_spread=0.4):
    """Synthesise ONE fixed variable-station batch in the exact packed format the model
    expects, by drawing per-sample station subsets and running ``variable_station_collate``.

    Returns ``(theta (B,6), context (B,W))`` torch tensors on CPU. Deterministic for a
    fixed ``seed`` so a fwd+bwd loop times pure compute on identical data.
    """
    import numpy as np
    import torch
    from seismo_sbi.sbi.compression.ML.dataloading import variable_station_collate

    rng = np.random.default_rng(seed)
    C = len(components)
    master_coords = np.stack([
        rng.uniform(station_center[0] - station_spread, station_center[0] + station_spread, n_master),
        rng.uniform(station_center[1] - station_spread, station_center[1] + station_spread, n_master),
    ], axis=1)

    items = []
    for _ in range(batch_size):
        frac = rng.uniform(*keep_fraction) if not isinstance(keep_fraction, (int, float)) else float(keep_fraction)
        n_keep = int(round(frac * n_master))
        n_keep = min(n_master, max(min(min_stations, n_master), n_keep))
        keep = np.sort(rng.choice(n_master, size=n_keep, replace=False))
        # Realistic-ish amplitude structure: per-station random scale so amplitude embedding /
        # SNR weighting exercise their real branches (not a degenerate constant array).
        scales = rng.uniform(0.1, 5.0, size=(n_keep, 1, 1))
        x_sub = torch.as_tensor(
            rng.standard_normal((n_keep, C, trace_length)) * scales, dtype=torch.float32)
        coords_sub = torch.as_tensor(master_coords[keep], dtype=torch.float32)
        source_vec = torch.as_tensor(
            np.asarray(source_fiducial[:n_cond]) + rng.normal(0, [0.05, 0.05, 2.0][:n_cond]),
            dtype=torch.float32)
        theta = torch.as_tensor(rng.standard_normal(6) * 0.5, dtype=torch.float32)
        items.append((theta, (x_sub, coords_sub, source_vec)))

    return variable_station_collate(items)


def default_augmentation_chain(sampling_rate, coda_prob=1.0):
    """The amplitude + time-shift + coda training-augmentation chain used by the benches."""
    nuisance = {
        "amplitude_error": [1.0],
        "time_shift_error": [1.0],
        "scattering_coda": [coda_prob],
    }
    stage = {k: "training_augmentation" for k in nuisance}
    effect_cfg = {
        "time_shift_error": {"uniform_offset": 1.0, "gaussian_sigma": 1.0},
        "scattering_coda": {"alpha_range": (0.2, 0.6)},
    }
    chain, params = build_augmentation_chain(
        nuisance, stage, effect_cfg, sampling_rate=sampling_rate,
    )
    return chain, params, nuisance
