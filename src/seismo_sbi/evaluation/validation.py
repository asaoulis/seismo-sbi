"""Held-out validation and TARP coverage for a trained model.

:func:`run_validation` takes the held-out tail of the simulation set (:func:`validation_dataset`),
samples the posterior for each simulation (:func:`sample_validation_posteriors`) and returns the
arrays for TARP coverage and recovery scatter; :func:`write_validation_outputs` writes the TARP
figure, the recovery scatter, example panels and the metrics JSON, one function each.
"""
from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def run_validation(
    sbi_pipeline,
    original_parameters,
    posterior,
    data_scaler,
    *,
    n_val: int = 200,
    n_show: int = 10,
    num_samples: int = 2000,
    device: Optional[str] = None,
    variable_stations: bool = True,
    cond_param_map: Optional[dict] = None,
) -> dict:
    """Posterior samples for the held-out tail of the simulation set.

    The last 10 % of the sorted simulations, at most ``n_val`` of them, are drawn with the
    training noise and augmentations and ``num_samples`` posterior samples taken for each.
    ``variable_stations`` (or a ``cond_param_map``, ``{block: [parameter names]}`` of the
    conditioning source parameters) packs the full station set as a variable-station context,
    conditioned on each simulation's true source; otherwise the flat data vector is the input.

    :returns: a dict of ``theta_phys`` and ``theta_scaled`` ``(n_val, n_dims)``,
        ``samples_phys`` and ``samples_scaled`` ``(num_samples, n_val, n_dims)`` (scaled to
        ``[0, 1]`` for TARP), ``show`` (the first ``n_show`` examples as ``InversionData``) and
        ``n_val``.
    """
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, val_idx = validation_dataset(sbi_pipeline, data_scaler, n_val=n_val)
    # Conditioned models are always variable-station (packed context).
    use_packed = bool(variable_stations) or (cond_param_map is not None)
    logger.info(f"  validation: {len(val_idx)} held-out sims (tail of {len(ds)}), "
                f"{num_samples} samples each; "
                f"conditioned={cond_param_map is not None}; "
                f"path={'packed (variable-station)' if use_packed else 'direct (fixed-station)'}.")

    theta_scaled_list, samples_scaled_list, samples_phys_list, shows = sample_validation_posteriors(
        sbi_pipeline, posterior, data_scaler, ds, val_idx, use_packed=use_packed, n_show=n_show,
        num_samples=num_samples, device=device, cond_param_map=cond_param_map)

    theta_scaled = np.stack(theta_scaled_list, axis=0)      # (n_val, n_dims)
    samples_scaled = np.stack(samples_scaled_list, axis=1)  # (num_samples, n_val, n_dims)
    theta_phys = data_scaler.inverse_transform(theta_scaled)
    samples_phys = np.stack(samples_phys_list, axis=1)      # (num_samples, n_val, n_dims)

    return {
        "theta_phys": theta_phys,
        "samples_phys": samples_phys,
        "theta_scaled": theta_scaled,
        "samples_scaled": samples_scaled,
        "show": shows,
        "n_val": len(val_idx),
    }


def validation_dataset(sbi_pipeline, data_scaler, *, n_val: int):
    """``(dataset, val_idx)``: the simulation set with the training noise and augmentations, and
    the indices of its held-out tail (the last 10 %, at most ``n_val``).
    """
    from seismo_sbi.sbi.npe.data.dataloading import TorchSimulationDataset
    from seismo_sbi.nuisance_effects.post_processing import build_augmentation_chain_from_parameters

    aug_chain, aug_params = build_augmentation_chain_from_parameters(
        sbi_pipeline.parameters,
        sampling_rate=sbi_pipeline.simulation_parameters.sampling_rate,
    )
    post_chain, post_params = build_augmentation_chain_from_parameters(
        sbi_pipeline.parameters,
        stage="training_augmentation_post_noise",
    )

    noise_sampler = getattr(sbi_pipeline, "training_noise_sampler", None)
    if noise_sampler is None:
        raise RuntimeError(
            "sbi_pipeline.training_noise_sampler is None; "
            "call build_eval_pipeline with the pipeline before run_validation."
        )

    ds = TorchSimulationDataset(
        data_loader=sbi_pipeline.data_manager.data_loader,
        data_folder=sbi_pipeline.simulations_output_path,
        parameter_name_map=sbi_pipeline.parameters.names,
        synthetic_noise_model_sampler=noise_sampler,
        data_scaler=data_scaler,
        augmentation_chain=aug_chain,
        augmentation_nuisance_params=aug_params,
        post_noise_augmentation_chain=post_chain,
        post_noise_nuisance_params=post_params,
    )

    n = len(ds)
    val_idx = list(range(int(0.90 * n), n))[:n_val]
    if not val_idx:
        raise RuntimeError(
            f"No held-out validation sims (n={n}, train_max_index={int(0.90*n)})."
        )
    return ds, val_idx


def sample_validation_posteriors(sbi_pipeline, posterior, data_scaler, ds, val_idx, *, use_packed, n_show,
                                 num_samples, device, cond_param_map):
    """Posterior samples for each held-out simulation ``ds[idx]``.

    :returns: lists over the simulations of the scaled truth, the scaled and physical samples,
        and the first ``n_show`` examples as ``InversionData``.
    """
    import torch
    from seismo_sbi.sbi.npe.source_conditioning import pack_subset_observation
    from seismo_sbi.sbi.types.results import InversionData

    data_loader = sbi_pipeline.data_manager.data_loader
    coords_all = np.asarray(ds.station_coords)  # (N_master, 2)

    def _sim_source_vec(sim_path):
        """Load the sim's TRUE stored source vector (for conditioned models)."""
        inp = data_loader.load_input_data(sim_path)
        return np.concatenate(
            [[inp[t][a] for a in attrs] for t, attrs in cond_param_map.items()]
        ).astype(float)

    theta_scaled_list: list = []
    samples_scaled_list: list = []
    samples_phys_list: list = []
    shows: list = []

    for k, idx in enumerate(val_idx):
        theta_s, x = ds[idx]
        theta_s_np = np.asarray(theta_s, dtype=float)

        # The paths differ only in how the observation becomes the model input. The packed path skips
        # transform(inverse_transform): the scale_shape clip is not invertible and would bias TARP.
        if use_packed:
            # Variable-station context over the full master set, conditioned on the sim's true source.
            sv = _sim_source_vec(ds.paths[idx]) if cond_param_map else None
            obs_in = pack_subset_observation(
                np.asarray(x), coords_all, source_vec=sv
            ).to(device)  # (1, W)
        else:
            # Direct path (fixed-station / unconditioned): the embedding net expects the
            # flat data vector.
            obs_in = torch.as_tensor(x, dtype=torch.float32).to(device).unsqueeze(0)

        s_scaled = posterior.sample(
            (num_samples,), obs_in, show_progress_bars=False
        ).cpu().numpy()  # (num_samples, n_dims) in [0,1]
        phys = data_scaler.inverse_transform(s_scaled)
        theta_scaled_list.append(theta_s_np)
        samples_scaled_list.append(s_scaled)
        samples_phys_list.append(phys)

        if k < n_show:
            theta_phys_k = data_scaler.inverse_transform(
                theta_s_np[np.newaxis, :]
            ).flatten()
            shows.append(
                InversionData(
                    theta0=theta_phys_k,
                    samples=phys,
                    data_scaler=data_scaler,
                )
            )
        if (k + 1) % 25 == 0:
            logger.info(f"    …{k + 1}/{len(val_idx)} val sims sampled")

    return theta_scaled_list, samples_scaled_list, samples_phys_list, shows


def write_validation_outputs(
    val: dict,
    out_dir,
    parameters,
    data_scaler,
    *,
    num_samples: int,
    conditioned: bool,
    n_show: int,
) -> dict:
    """Figures and ``evaluation_metrics.json`` in ``out_dir`` from a :func:`run_validation` dict.

    Writes ``tarp_coverage.png`` (TARP expected coverage), ``recovery_scatter.svg`` (true
    against recovered gamma, delta and Mw), ``n_show`` example panels under ``examples/`` and
    the metrics JSON, which records ``num_samples`` and ``conditioned`` with the metrics.
    ``parameters`` are the pipeline's parameters, for the panel labels.

    :returns: the dict written to ``evaluation_metrics.json``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_scaled = val["theta_scaled"]
    samples_scaled = val["samples_scaled"]
    theta_phys = val["theta_phys"]
    samples_phys = val["samples_phys"]
    shows = val.get("show", [])
    n_val = val.get("n_val", theta_scaled.shape[0])

    figures: dict = {}
    ecp, alpha = write_tarp_figure(theta_scaled, samples_scaled, out_dir, figures)
    write_recovery_scatter(theta_phys, samples_phys, out_dir, figures)
    if shows:
        write_example_panels(shows, n_show, parameters, data_scaler, out_dir, figures)
    return write_metrics_json(val, ecp, alpha, out_dir, figures, n_val=n_val, num_samples=num_samples,
                              conditioned=conditioned)


def write_tarp_figure(theta_scaled, samples_scaled, out_dir, figures):
    """TARP expected coverage ``(ecp, alpha)``, drawn to ``tarp_coverage.png`` and entered in
    ``figures``; ``(None, None)`` when it cannot be computed.
    """
    from seismo_sbi.evaluation import posterior_metrics
    from seismo_sbi.plotting.coverage import plot_coverage

    ecp = alpha = None
    try:
        ecp, alpha = posterior_metrics.tarp_coverage(samples_scaled, theta_scaled, num_bootstrap=100)
        cov_path = out_dir / "tarp_coverage.png"
        plot_coverage(
            {"NPE ML": (ecp, alpha)},
            colors=["#6495ED"],
            savefig=str(cov_path),
            title="TARP expected coverage",
        )
        figures["tarp_coverage"] = str(cov_path)
        logger.info(f"    wrote TARP coverage -> {cov_path}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"    [warn] TARP coverage failed: {type(e).__name__}: {e}")
    return ecp, alpha


def write_recovery_scatter(theta_phys, samples_phys, out_dir, figures):
    """True against recovered gamma, delta and Mw, drawn to ``recovery_scatter.svg`` and entered
    in ``figures``.
    """
    from seismo_sbi.plotting import evaluation as ev

    try:
        sc_path = out_dir / "recovery_scatter.svg"
        ev.plot_recovery_scatter(theta_phys, samples_phys, figsave=sc_path)
        figures["recovery_scatter"] = str(sc_path)
        logger.info(f"    wrote recovery scatter -> {sc_path}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"    [warn] recovery scatter failed: {type(e).__name__}: {e}")


def write_example_panels(shows, n_show, parameters, data_scaler, out_dir, figures):
    """Posterior panels for the first ``n_show`` examples under ``out_dir/examples``."""
    from seismo_sbi.plotting.results_plotting import SBIPipelinePlotter

    try:
        vp = SBIPipelinePlotter(str(out_dir), parameters)
        vp.initialise_posterior_plotter(
            data_scaler,
            parameters.parameter_to_vector("information")[:6],
        )
        for i, invdata in enumerate(shows[:n_show]):
            try:
                vp.plot_chain_consumer(
                    "examples",
                    f"val_{i:02d}",
                    {"NPE ML": invdata},
                    kde=True,
                    savefig=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"    [warn] val example {i} failed: {type(e).__name__}: {e}")
        figures["examples_dir"] = str(out_dir / "examples")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"    [warn] example panels failed: {type(e).__name__}: {e}")


def write_metrics_json(val, ecp, alpha, out_dir, figures, *, n_val, num_samples, conditioned):
    """The evaluation metrics of ``val`` with the run's settings and ``figures``, written to
    ``evaluation_metrics.json``.
    """
    from seismo_sbi.evaluation import posterior_metrics

    metrics: dict = {}
    try:
        metrics = posterior_metrics.compute_evaluation_metrics(val, ecp=ecp, alpha=alpha)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"    [warn] metric computation failed: {type(e).__name__}: {e}")

    result = {
        "n_val": n_val,
        "num_samples": num_samples,
        "conditioned": conditioned,
        "figures": figures,
        "metrics": metrics,
    }
    with open(out_dir / "evaluation_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
