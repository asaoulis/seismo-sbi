"""Posterior predictive checks: re-simulate posterior draws and score them against the data.

:class:`PosteriorPredictiveChecks` simulates an ensemble per method, selects the best synthetics,
and evaluates registered misfit metrics against the observation, by default those of
:mod:`seismo_sbi.evaluation.ppc_metrics`; :func:`plot_metric_bars` compares the metrics across methods.
"""

import traceback
import warnings
from collections import OrderedDict
from typing import Callable, Dict, Iterable, List, Optional, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from seismo_sbi.evaluation import ppc_metrics
from seismo_sbi.nuisance_effects.post_processing import PostProcessingChain
from seismo_sbi.simulators.simulation_io import component_alias
from seismo_sbi.utils.parallel import tqdm_joblib


class PosteriorPredictiveChecks:
    """
    Extensible PPC runner.

    Parameters
    ----------
    simulator : callable
        Callable(param_dict) -> 1D numpy array synthetic (same length as observation).
    covariance_matrix : optional object
        Noise covariance for the reduced chi-square: an object whose
        ``compute_loss(residual, reduce=True)`` returns ``-chi2 / 2``. Without one the chi-square
        is ``dot(r, r)``.
    receivers : optional
        Object with .iterate() yielding receivers where each receiver has .components,
        used to infer per-trace shapes.
    n_traces : optional int
        Provide if receivers is not available.
    sample_rate : optional float
        Sampling rate (Hz). Required for PSD / band-power metrics.
    n_jobs : int
        Number of parallel jobs for simulation.
    dof_override : optional int
        If provided, use as degree-of-freedom for reduced chi^2 calculation.
    augmentation_chains : optional dict
        Maps ``ensemble_name -> PostProcessingChain`` of Category-2 nuisance
        effects (the SAME effects used in training-time augmentation) to fold
        into each synthetic of that ensemble. A time shift is a ``TimeShiftErrorEffect`` in
        the chain.
    augmentation_nuisance_params : optional dict
        Maps ``ensemble_name -> {nuisance_key: value}`` activating the effects
        in that ensemble's chain (e.g. ``{"time_shift_error": 1.0}``).  Missing
        entries default to ``{}`` (effects inactive / identity).
    """

    def __init__(
        self,
        simulator: Callable,
        covariance_matrix: Optional[object] = None,
        receivers: Optional[object] = None,
        n_traces: Optional[int] = None,
        sample_rate: Optional[float] = 1.0,
        n_jobs: int = 20,
        dof_override: Optional[int] = None,
        augmentation_chains: Optional[Dict[str, PostProcessingChain]] = None,
        augmentation_nuisance_params: Optional[Dict[str, dict]] = None,
        max_shift_samples: Optional[int] = None,
        autocorr_maxlag: int = 50,
    ):
        self.simulator = simulator
        self.covariance_matrix = covariance_matrix
        self.receivers = receivers
        self.n_jobs = int(n_jobs)
        self.sample_rate = sample_rate
        self.dof_override = dof_override
        #: Lag budget (samples, each direction) for "Shifted corr misfit".  ``None`` ->
        #: 10 % of the trace length, i.e. scaled to whatever window the traces span.
        self.max_shift_samples = max_shift_samples
        #: Max lag (samples) for the autocorrelation-shape metric.
        self.autocorr_maxlag = int(autocorr_maxlag)

        # infer n_traces
        if n_traces is not None:
            self.n_traces = int(n_traces)
        elif receivers is not None:
            # receivers.iterate() expected to be generator
            self.n_traces = sum(len(r.components) for r in receivers.iterate())
        else:
            self.n_traces = None

        # store per-ensemble nuisance augmentation chains + their activation params
        # mapping ensemble_name -> PostProcessingChain / {nuisance_key: value}
        self.augmentation_chains = augmentation_chains or {}
        self.augmentation_nuisance_params = augmentation_nuisance_params or {}

        # metric registry: name -> callable(obs, synthetics, meta) -> per-sample np.array
        self.metrics = OrderedDict()
        # register defaults
        self.add_metric(r"$\\chi^2$", ppc_metrics.reduced_chi2_misfit)
        self.add_metric("Correlation misfit", ppc_metrics.correlation_misfit_per_sample)
        self.add_metric("Power misfit", ppc_metrics.power_misfit)
        self.add_metric("Envelope misfit", ppc_metrics.envelope_misfit)
        self.add_metric("MSE", ppc_metrics.mse)
        # Shift-tolerant pair: judge waveform shape with the arrival free, so a mechanism is not
        # charged for travel-time error from the 1-D model and hypocentre.
        self.add_metric("Shifted corr misfit", ppc_metrics.shifted_correlation_misfit)
        self.add_metric("Autocorr misfit", ppc_metrics.autocorrelation_misfit)

    # --- Public API ---
    def add_metric(self, name: str, func: Callable):
        """
        Register a new metric. func(obs, synthetics, meta) -> np.ndarray (len = n_synthetics)
        meta is a dict with helpful entries, e.g. sample_rate, n_traces, covariance_matrix, dof_override
        """
        if name in self.metrics:
            raise KeyError(f"Metric {name} already exists")
        self.metrics[name] = func

    def _metric_meta(self):
        """The ``meta`` dict every metric receives (see :mod:`seismo_sbi.evaluation.ppc_metrics`)."""
        return {
            "sample_rate": self.sample_rate,
            "n_traces": self.n_traces,
            "covariance_matrix": self.covariance_matrix,
            "dof_override": self.dof_override,
            "max_shift_samples": self.max_shift_samples,
            "autocorr_maxlag": self.autocorr_maxlag,
        }

    def simulate_ensemble(self, param_list: List[dict], num_samples: int = 100, *, ensemble_name: Optional[str] = None):
        """
        Simulate 'num_samples' synthetics by drawing from param_list (without mutating input).
        Returns array shape (num_samples, data_length)
        """
        synthetics = [] 
        inversion_data =param_list 
        samples = inversion_data.samples 
        np.random.shuffle(samples) 
        samples = samples[:num_samples] 
        with tqdm_joblib(tqdm(desc="Simulating synthetics for PPCs", total=len(samples))):
             with joblib.parallel_backend('loky', n_jobs=self.n_jobs):
                results = joblib.Parallel()( 
                     joblib.delayed(self.simulator)(param_dict) for param_dict in samples 
                )
                # Fold in per-ensemble nuisance augmentation (same effects as training).
                synthetics.extend(self._apply_augmentation(results, ensemble_name))
        return np.vstack(synthetics)

    def _apply_augmentation(self, results, ensemble_name):
        """Apply the per-ensemble nuisance augmentation chain to a list of flat synthetics.

        Returns the (possibly augmented) list of 1-D vectors. A no-op when no chain is
        configured for ``ensemble_name`` (or no receivers). If a chain raises on a given
        vector the original (un-augmented) vector is kept, but the failure is surfaced via
        a warning rather than silently swallowed.
        """
        chain = self.augmentation_chains.get(ensemble_name) if ensemble_name is not None else None
        if self.receivers is None or chain is None or not chain.effects:
            return list(results)

        print("Applying nuisance augmentation chain for ensemble:", ensemble_name)
        nuisance_params = self.augmentation_nuisance_params.get(ensemble_name, {})
        augmented = []
        for vec in results:
            try:
                outputs_map = self._vec_to_outputs_map(vec)
                processed = chain(outputs_map, self.receivers, nuisance_params)
                augmented.append(self._outputs_map_to_vec(processed))
            except Exception as exc:
                warnings.warn(
                    f"PPC nuisance augmentation failed for ensemble {ensemble_name!r}; "
                    f"using the un-augmented synthetic instead: {exc!r}"
                )
                augmented.append(vec)
        return augmented

    def simulate_ensembles(
        self,
        ensemble_params_dict: Dict[str, List[dict]],
        num_samples: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """Simulate synthetics for each ensemble without any selection or metrics.

        Parameters
        ----------
        ensemble_params_dict : Dict[str, List[dict]]
            Mapping ensemble-name -> list-like container of parameter dicts.
        num_samples : int, optional
            Number of synthetics to simulate per ensemble.

        Returns
        -------
        ensembles_synthetics : dict
            ensembles_synthetics[name] -> np.ndarray, shape (num_samples, data_len)
        """
        ensembles_synthetics: Dict[str, np.ndarray] = {}
        for name, params in ensemble_params_dict.items():
            ensembles_synthetics[name] = self.simulate_ensemble(
                params, num_samples=num_samples, ensemble_name=name
            )
        return ensembles_synthetics

    def select_best_synthetics(
        self,
        observation: np.ndarray,
        ensembles_synthetics: Dict[str, np.ndarray],
        num_selected_samples: Optional[int] = None,
        selection_metric: str = r"$\\chi^2$",
    ):
        """Select best-matching synthetics for each ensemble given precomputed synthetics.

        This does **not** run any simulations or compute summary metrics, it just
        applies the chosen selection metric on the raw synthetics and returns a
        dictionary with the selected subsets.

        Parameters
        ----------
        observation : np.ndarray
            Observed data vector.
        ensembles_synthetics : Dict[str, np.ndarray]
            Mapping ensemble-name -> synthetics array of shape (n_samples, data_len).
        num_selected_samples : int, optional
            If provided and > 0, selects up to this many best-matching synthetics
            per ensemble. If None or <= 0, returns the input unchanged.
        selection_metric : str, optional
            Name of the metric (registered via ``add_metric``) to minimize.

        Returns
        -------
        selected_synthetics : dict
            selected_synthetics[name] -> np.ndarray of shape (k, data_len)
        selection_indices : dict
            selection_indices[name] -> np.ndarray of indices into the original
            synthetics array, or None if no selection was applied.
        """
        obs = np.asarray(observation).ravel()
        meta = self._metric_meta()

        selected_synthetics: Dict[str, np.ndarray] = {}
        selection_indices: Dict[str, Optional[np.ndarray]] = {}

        for name, synthetics_full in ensembles_synthetics.items():
            if num_selected_samples is not None and num_selected_samples > 0:
                syn_sel, idx = self._select_best_synthetics(
                    obs,
                    synthetics_full,
                    meta,
                    metric_name=selection_metric,
                    k=num_selected_samples,
                )
            else:
                syn_sel, idx = synthetics_full, None

            selected_synthetics[name] = syn_sel
            selection_indices[name] = idx

        return selected_synthetics, selection_indices

    def evaluate_metrics(
        self,
        observation: np.ndarray,
        ensembles_synthetics: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, dict]]:
        """Compute all registered metrics for each ensemble on given synthetics.

        Parameters
        ----------
        observation : np.ndarray
            Observed data vector.
        ensembles_synthetics : Dict[str, np.ndarray]
            Mapping ensemble-name -> synthetics array used for metric evaluation.

        Returns
        -------
        results : dict
            results[name] -> dict[metric_name] -> {"mean", "var", "raw"}
        """
        obs = np.asarray(observation).ravel()
        meta = self._metric_meta()

        results: Dict[str, Dict[str, dict]] = {}
        for name, synthetics in ensembles_synthetics.items():
            if synthetics.size == 0:
                results[name] = {
                    m: {"mean": np.nan, "var": np.nan, "raw": np.array([])}
                    for m in self.metrics
                }
                continue

            metrics_out: Dict[str, dict] = {}
            for mname, mfunc in self.metrics.items():
                try:
                    vals = np.asarray(mfunc(obs, synthetics, meta), dtype=float)
                    if vals.ndim != 1 or vals.shape[0] != synthetics.shape[0]:
                        raise ValueError(
                            f"Metric {mname} must return 1D array length n_synthetics"
                        )
                except Exception:
                    vals = np.full((synthetics.shape[0],), np.nan, dtype=float)
                metrics_out[mname] = {
                    "mean": float(np.nanmean(vals)),
                    "var": float(np.nanvar(vals)),
                    "raw": vals,
                }

            results[name] = metrics_out

        return results

    def run_posterior_predictive_metrics(
        self,
        observation: np.ndarray,
        ensemble_params_dict: Dict[str, List[dict]],
        num_samples: int = 1000,
        num_selected_samples: Optional[int] = None,
        selection_metric: str = r"$\\chi^2$",
    ):
        """End-to-end PPC pipeline: simulate -> (optionally) select -> evaluate metrics.

        This is a convenience wrapper around ``simulate_ensembles``,
        ``select_best_synthetics`` and ``evaluate_metrics``. For experiments
        where you want to reuse the same simulated synthetics across different
        selection criteria or metrics, call those methods directly instead of
        this wrapper.
        """
        # 1) simulate full ensembles
        ensembles_synthetics_full = self.simulate_ensembles(
            ensemble_params_dict, num_samples=num_samples
        )

        # 2) optionally select a subset of best-matching synthetics
        ensembles_synthetics_sel, selection_indices = self.select_best_synthetics(
            observation,
            ensembles_synthetics_full,
            num_selected_samples=num_selected_samples,
            selection_metric=selection_metric,
        )

        # 3) evaluate metrics on the (possibly) selected synthetics
        results = self.evaluate_metrics(observation, ensembles_synthetics_sel)

        # Optionally attach selection indices for downstream inspection
        for name, idx in selection_indices.items():
            if idx is not None:
                results[name]["_selection_indices"] = {
                    "mean": np.nan,
                    "var": np.nan,
                    "raw": idx,
                }

        return results, ensembles_synthetics_sel

    # --- Time-shift helpers ---
    def _vec_to_outputs_map(self, vec: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Reshape flattened vector into {station: {component: trace}} according to receivers order."""
        if self.receivers is None or self.n_traces is None:
            raise ValueError("Receivers and n_traces are required to apply time shifts.")
        vec = np.asarray(vec).ravel()
        # compute trace length per component from ordering implied by receivers
        total_traces = self.n_traces
        if len(vec) % total_traces != 0:
            raise ValueError("Vector length not divisible by number of traces.")
        tlen = len(vec) // total_traces
        outputs = {}
        idx = 0
        for rec in self.receivers.iterate():
            station = rec.station_name
            outputs[station] = {}
            for comp in rec.components:
                outputs[station][comp] = vec[idx * tlen : (idx + 1) * tlen]
                idx += 1
        return outputs

    def _outputs_map_to_vec(self, outputs: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """Flatten {station: {component: trace}} back to 1D vector in receivers/components order."""
        parts = []
        for rec in self.receivers.iterate():
            station = rec.station_name
            for comp in rec.components:
                trace = outputs[station].get(comp)
                if trace is None:
                    trace = outputs[station].get(component_alias(comp))
                if trace is None:
                    raise KeyError(f"Missing trace for {station}:{comp}")
                parts.append(np.asarray(trace))
        return np.concatenate(parts)

    # --- Best-synthetic selection ---
    def _select_best_synthetics(
        self,
        obs: np.ndarray,
        synthetics: np.ndarray,
        meta: dict,
        metric_name: str,
        k: int,
    ):
        """Select the k best-matching synthetics according to a registered metric.

        The metric must be present in ``self.metrics`` and is interpreted as
        "lower is better" for the purpose of selection.

        Returns
        -------
        synthetics_subset : np.ndarray
            Array of shape (k', data_len) where k' = min(k, n_synthetics).
        indices : np.ndarray
            Indices (into the original "synthetics" array) of the selected samples.
        """
        n_total = synthetics.shape[0]
        if n_total == 0 or k <= 0:
            return synthetics[:0], np.empty((0,), dtype=int)

        k = min(k, n_total)

        # fall back to no selection if the metric is not registered
        if metric_name not in self.metrics:
            print("Using fallback - something wrong with metric name or metric computation, returning first k samples. Metric name:", metric_name)
            return synthetics[:k], np.arange(k, dtype=int)

        metric_func = self.metrics[metric_name]
        try:
            vals = np.asarray(metric_func(obs, synthetics, meta), dtype=float)
            if vals.ndim != 1 or vals.shape[0] != n_total:
                raise ValueError("selection metric must return 1D array with length = n_synthetics")
        except Exception:
            # if metric computation fails, just return first k
            print("Metric computation failed during selection, returning first k samples. Error:")
            traceback.print_exc()
            return synthetics[:k], np.arange(k, dtype=int)
        # lower is better
        order = np.argsort(vals)
        best_idx = order[:k]
        return synthetics[best_idx], best_idx


def plot_metric_bars(
    metrics_dict: Dict[str, Dict[str, float]],
    higher_is_better: Optional[Union[Iterable[str], Dict[str, bool]]] = None,
    title: Optional[str] = None,
    figsize=(15, 8),
    annotate: bool = False,
    rotation: int = 20,
    include_substring: Optional[str] = None,
    exclude_substrings: Iterable[str] = ("var",),
):
    """
    Plot grouped bars of approaches across metrics, normalized so the best per metric == 1.
    For lower-is-better metrics: min/value (best = 1, others <= 1)
    For higher-is-better metrics: value/max (best = 1, others <= 1)
    """
    first_metrics = next(iter(metrics_dict.values()))
    columns_order = list(first_metrics.keys())

    df = pd.DataFrame(metrics_dict).T.astype(float)
    cols = [c for c in columns_order if c in df.columns] + [c for c in df.columns if c not in columns_order]

    def keep_metric(name: str) -> bool:
        n = name.lower()
        if include_substring is not None and include_substring.lower() not in n:
            return False
        if any(s.lower() in n for s in (exclude_substrings or [])):
            return False
        return True

    metric_cols = [c for c in cols if keep_metric(c)]
    if not metric_cols:
        raise ValueError("No metrics left after applying include/exclude filters.")

    df = df[metric_cols]

    if higher_is_better is None:
        hib = {m: False for m in df.columns}
    elif isinstance(higher_is_better, dict):
        hib = {m: bool(higher_is_better.get(m, False)) for m in df.columns}
    else:
        hib = {m: (m in set(higher_is_better)) for m in df.columns}

    norm_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for m in df.columns:
        col = df[m].astype(float).replace([np.inf, -np.inf], np.nan)
        if hib[m]:  # higher is better
            best = np.nanmax(col.values)
            norm_df[m] = col / best if best != 0 else np.nan
        else:  # lower is better
            best = np.nanmin(col.values)
            norm_df[m] = col/best if np.all(col != 0) else np.nan
    approaches = list(norm_df.index)
    metrics = list(norm_df.columns)
    n_metrics = len(metrics)
    n_approaches = len(approaches)

    x = np.arange(n_metrics)
    total_width = 0.8
    bar_width = total_width / max(n_approaches, 1)
    offset_start = -total_width / 2 + bar_width / 2

    fig, ax = plt.subplots(figsize=figsize)

    for i, approach in enumerate(approaches):
        y = norm_df.loc[approach, metrics].values
        ax.bar(x + offset_start + i * bar_width, y, width=bar_width, label=approach, alpha=0.9)
        if annotate:
            for xi, yi in zip(x + offset_start + i * bar_width, y):
                ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=rotation, ha="right")
    ax.set_ylim(0.9, 1.2)
    ax.set_ylabel("Normalized score (best per metric = 1)")
    ax.set_xlabel("Posterior Predictive Metric")

    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.7)
    ax.legend(title="Approach", frameon=False, ncol=min(n_approaches, 3))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    plt.tight_layout()
    return fig, ax

def extract_means(metrics_dict):
    """
    Flatten nested metrics_dict that have subdicts like {'mean': x, 'var': y, ...}
    into a simple {approach: {metric_mean_name: value}} structure.
    """
    out = {}
    for approach, metrics in metrics_dict.items():

        flat = {}
        for name, val in metrics.items():
            # remove double backslashes from metric names (e.g. from LaTeX formatting)
            name = name.replace(r'$', r'')
            name = name.replace(r'\\', r'')
            name = name.replace(r'^', r'')
            if "selection" in name.lower():
                continue
            if isinstance(val, dict):
                # If it has a 'mean' entry, store it as 'metric_mean'
                if "mean" in val:
                    flat[f"{name}_mean"] = val["mean"]
            else:
                # Already scalar
                flat[name] = val
        out[approach] = flat
    return out
