"""
validation.py
=============
The ONE generic validation / TARP engine that subsumes BOTH:

  * ``scripts/continuity/_eval_inference.evaluate_validation_set``
    (fixed-station / unconditioned LV2 path), and
  * ``scripts/santorini_pathbreaker/run_posttrain_eval._validation_tarp``
    (variable-station / optionally-conditioned Santorini path).

The two previous implementations diverged **only** in how a held-out sim's
observation is turned into a posterior sample; everything downstream (TARP,
recovery scatter, example panels, metrics JSON) is identical and lives in
``seismo_sbi.plotting.evaluation`` / ``plotting.coverage``.

Public API
----------
``run_validation(...)`` → dict
    Draw the held-out validation tail, run posterior inference on each sim, and
    return arrays for TARP coverage + recovery scatter plus a handful of
    per-example InversionData objects.

``write_validation_outputs(val, out_dir, ...)`` → dict
    Write figures + metrics JSON from the ``run_validation`` return value.

Heavy deps (torch, seismo_sbi pipeline classes) are imported lazily inside
each function so importing this module during the fast unit-test gate is cheap.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


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
    """Draw the held-out validation set and run posterior inference on each sim.

    Builds a plain (full-station, unconditioned) ``TorchSimulationDataset`` with
    the same augmentation chains and training noise the model trained on, takes
    the last 10% tail of the sorted sim files as the held-out split (matching
    ``train_NPE.py``), then samples the posterior for each sim. The two model
    families differ ONLY in how the observation is packed into the model's input:

    * **Variable-station / conditioned** (``variable_stations`` or ``cond_param_map``):
      pack the full master station set via ``pack_subset_observation`` (the inference
      mirror of the training collate); for a conditioned model the sim's TRUE stored
      ``source_location`` is fed as ``source_vec`` (matches how real events feed the
      catalogue location at inference; conditioning noise is train-only, not applied here).
    * **Fixed-station / unconditioned**: feed the flat data vector straight to the
      posterior, reproducing the old ``evaluate_validation_set`` direct path.

    Both paths keep the RAW scaled posterior draws for TARP and ``inverse_transform``
    once for physical units — no lossy ``transform(inverse_transform(.))`` round-trip.

    Parameters
    ----------
    sbi_pipeline:
        A fully-loaded ``SingleEventPipeline`` (as returned by
        ``build_eval_pipeline``).
    original_parameters:
        Deep-copy of pipeline parameters snapshotted before compressor build.
    posterior:
        The trained ML posterior (e.g. from ``build_ml_posterior``).
    data_scaler:
        FlexibleScaler instance (from ``build_flexible_scaler``).
    n_val:
        Maximum number of validation sims to evaluate.
    n_show:
        Number of per-example InversionData objects to include in the output.
    num_samples:
        Posterior samples to draw per validation sim.
    device:
        ``"cuda"`` / ``"cpu"`` (auto-detected if None).
    variable_stations:
        ``True`` (default) → pack the observation via ``pack_subset_observation``
        (variable-station models). Also forced ``True`` when ``cond_param_map`` is
        set (conditioned models are always variable-station). ``False`` AND
        ``cond_param_map`` is ``None`` → feed the flat ``(N·C·T,)`` data vector
        directly, correct for fixed-station / unconditioned models (e.g. the LV2
        CNN/TCN checkpoints).
    cond_param_map:
        ``ml_conditioning.param_map`` dict (e.g.
        ``{"source_location": ["latitude", "longitude", "depth"]}``).
        If set, each sim's TRUE stored source vector is fed as conditioning.
        ``None`` → unconditioned model.

    Returns
    -------
    dict with keys:
        ``theta_phys``    (n_val, n_dims) — truth in physical units
        ``samples_phys``  (num_samples, n_val, n_dims) — samples in physical units
        ``theta_scaled``  (n_val, n_dims) — truth in [0,1] (for TARP)
        ``samples_scaled`` (num_samples, n_val, n_dims) — samples in [0,1]
        ``show``          list[InversionData] — first n_show examples (physical)
        ``n_val``         int — actual number of sims evaluated
    """
    import torch
    from seismo_sbi.sbi.compression.ML.dataloading import TorchSimulationDataset
    from seismo_sbi.sbi.compression.ML.source_conditioning import pack_subset_observation
    from seismo_sbi.instaseis_simulator.post_processing import (
        build_augmentation_chain_from_parameters,
    )
    from seismo_sbi.sbi.types.results import InversionData

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mirror training: same augmentation chains so the held-out eval distribution
    # matches what the posterior was trained on.
    aug_chain, aug_params = build_augmentation_chain_from_parameters(
        sbi_pipeline.parameters,
        sampling_rate=sbi_pipeline.simulation_parameters.sampling_rate,
    )
    post_chain, post_params = build_augmentation_chain_from_parameters(
        sbi_pipeline.parameters,
        stage="training_augmentation_post_noise",
    )

    # Plain dataset — no station subsampler, no conditioning — so __getitem__
    # returns the full (N,C,T) noisy+augmented obs + scaled theta.  Model-correct
    # packing (variable-station via pack_subset_observation, or the flat fixed-station
    # vector) happens per-sim in the loop below.
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
    data_loader = sbi_pipeline.data_manager.data_loader
    coords_all = np.asarray(ds.station_coords)  # (N_master, 2)

    n = len(ds)
    val_idx = list(range(int(0.90 * n), n))[:n_val]
    if not val_idx:
        raise RuntimeError(
            f"No held-out validation sims (n={n}, train_max_index={int(0.90*n)})."
        )
    # Conditioned models are always variable-station (packed context).
    use_packed = bool(variable_stations) or (cond_param_map is not None)
    print(
        f"  validation: {len(val_idx)} held-out sims (tail of {n}), "
        f"{num_samples} samples each; "
        f"conditioned={cond_param_map is not None}; "
        f"path={'packed (variable-station)' if use_packed else 'direct (fixed-station)'}."
    )

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

        # Both paths differ ONLY in how the held-out observation is turned into the
        # model's input tensor; both then keep the RAW scaled posterior draws (for TARP)
        # and inverse_transform once for physical units (for the recovery scatter). The
        # packed branch deliberately does NOT round-trip through transform(inverse_transform)
        # — under the scale_shape MT scaler that clipping is non-invertible and would make
        # the conditioned-path TARP curve inconsistent with the direct path.
        if use_packed:
            # Packed path (variable-station, optionally conditioned): the embedding net
            # expects the 2-D variable-station context. Full master set, no dropout; for a
            # conditioned model feed the sim's TRUE stored source vector.
            sv = _sim_source_vec(ds.paths[idx]) if cond_param_map else None
            obs_in = pack_subset_observation(
                np.asarray(x), coords_all, source_vec=sv
            ).to(device)  # (1, W)
        else:
            # Direct path (fixed-station / unconditioned): the embedding net expects the
            # flat data vector (mirrors _eval_inference.evaluate_validation_set).
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
            print(f"    …{k + 1}/{len(val_idx)} val sims sampled")

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
    """Write figures + metrics JSON from the ``run_validation`` output dict.

    Extracted from the figure/metrics tail of ``run_posttrain_eval._validation_tarp``
    and ``visualise_results.py`` Stage B.  Reuses the existing src plotting helpers
    wholesale — no new plotting code.

    Produces:
        ``tarp_coverage.png``      — TARP expected-coverage curve
        ``recovery_scatter.svg``   — true-vs-recovered gamma/delta/Mw
        ``examples/``              — n_show per-example MT/nodal corner panels
        ``evaluation_metrics.json``— TARP calibration error, empirical coverage, FoM, …

    Parameters
    ----------
    val:
        Return value of ``run_validation``.
    out_dir:
        Directory to write outputs into (e.g. ``layout.validation_dir()``).
    parameters:
        Pipeline parameters (original_parameters from build_eval_pipeline).
    data_scaler:
        FlexibleScaler (same as passed to run_validation).
    num_samples:
        Number of posterior samples used (for provenance in the JSON).
    conditioned:
        Whether the model is conditioned (for provenance in the JSON).
    n_show:
        Number of example panels to render.

    Returns
    -------
    dict  — the metrics dict written to ``evaluation_metrics.json``.
    """
    from seismo_sbi.plotting import evaluation as ev
    from seismo_sbi.plotting.coverage import plot_coverage
    from seismo_sbi.plotting.results_plotting import SBIPipelinePlotter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    theta_scaled = val["theta_scaled"]
    samples_scaled = val["samples_scaled"]
    theta_phys = val["theta_phys"]
    samples_phys = val["samples_phys"]
    shows = val.get("show", [])
    n_val = val.get("n_val", theta_scaled.shape[0])

    figures: dict = {}
    ecp = alpha = None

    # ── TARP coverage ──────────────────────────────────────────────────────────
    try:
        ecp, alpha = ev.tarp_coverage(samples_scaled, theta_scaled, num_bootstrap=100)
        cov_path = out_dir / "tarp_coverage.png"
        plot_coverage(
            {"NPE ML": (ecp, alpha)},
            colors=["#6495ED"],
            savefig=str(cov_path),
            title="TARP expected coverage",
        )
        figures["tarp_coverage"] = str(cov_path)
        print(f"    wrote TARP coverage -> {cov_path}")
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] TARP coverage failed: {type(e).__name__}: {e}")

    # ── Recovery scatter ───────────────────────────────────────────────────────
    try:
        sc_path = out_dir / "recovery_scatter.svg"
        ev.plot_recovery_scatter(theta_phys, samples_phys, figsave=sc_path)
        figures["recovery_scatter"] = str(sc_path)
        print(f"    wrote recovery scatter -> {sc_path}")
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] recovery scatter failed: {type(e).__name__}: {e}")

    # ── Example panels ─────────────────────────────────────────────────────────
    if shows:
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
                    print(f"    [warn] val example {i} failed: {type(e).__name__}: {e}")
            figures["examples_dir"] = str(out_dir / "examples")
        except Exception as e:  # noqa: BLE001
            print(f"    [warn] example panels failed: {type(e).__name__}: {e}")

    # ── Metrics JSON ──────────────────────────────────────────────────────────
    metrics: dict = {}
    try:
        metrics = ev.compute_evaluation_metrics(val, ecp=ecp, alpha=alpha)
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] metric computation failed: {type(e).__name__}: {e}")

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
