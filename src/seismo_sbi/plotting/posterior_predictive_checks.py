import numpy as np
import contextlib
import joblib
from tqdm import tqdm
from typing import Callable, Dict, List, Optional
from collections import OrderedDict

# Optional seismology helpers
try:
    from scipy.signal import hilbert, welch
except Exception:
    hilbert = None
    welch = None

from seismo_sbi.instaseis_simulator.utils import apply_station_time_shifts


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar.
    Robust to exceptions and restores original method on exit.
    """
    original = joblib.parallel.Parallel.print_progress

    def print_progress(self):
        try:
            # self.n_completed_tasks exists on joblib >=0.14
            completed = getattr(self, "n_completed_tasks", None)
            if completed is None:
                return original(self)
            # update by difference
            delta = int(completed - getattr(tqdm_object, "n", 0))
            if delta > 0:
                tqdm_object.update(n=delta)
        except Exception:
            # fallback to original if anything goes wrong
            try:
                original(self)
            except Exception:
                pass

    joblib.parallel.Parallel.print_progress = print_progress
    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original
        tqdm_object.close()


class PosteriorPredictiveChecks:
    """
    Extensible PPC runner.

    Parameters
    ----------
    simulator : callable
        Callable(param_dict) -> 1D numpy array synthetic (same length as observation).
    covariance_matrix : optional object
        Optional covariance wrapper used to compute Mahalanobis distance.
        Supported (attempted) interfaces (in order):
          - obj.solve(rhs) -> C^{-1} rhs  OR obj.apply_inverse(vec)
            then chi2 = r^T (C^{-1} r)
          - obj.compute_loss(residual, reduce=True) returning either
              * -0.5 * chi2   (common in some codebases), or
              * 0.5 * chi2
            we try to infer sign/scale, but fallback to dot(r, r).
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
    random_shift_distributions : optional dict
        If provided, enables per-ensemble station time shifts using the same
        convention as TorchSimulationDataset: a global integer shift drawn
        from N(mean, std=mean) rounded, plus a per-station uniform integer
        jitter in [-half_range, half_range].
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
        random_shift_distributions: Optional[Dict[str, tuple]] = None,
    ):
        self.simulator = simulator
        self.covariance_matrix = covariance_matrix
        self.receivers = receivers
        self.n_jobs = int(n_jobs)
        self.sample_rate = sample_rate
        self.dof_override = dof_override

        # infer n_traces
        if n_traces is not None:
            self.n_traces = int(n_traces)
        elif receivers is not None:
            # receivers.iterate() expected to be generator
            self.n_traces = sum(len(r.components) for r in receivers.iterate())
        else:
            self.n_traces = None

        # store per-ensemble random shift distributions dict
        # mapping ensemble_name -> (mean, half_range)
        self.random_shift_distributions = random_shift_distributions or {}

        # metric registry: name -> callable(obs, synthetics, meta) -> per-sample np.array
        self.metrics = OrderedDict()
        # register defaults
        self.add_metric(r"$\\chi^2$", self._metric_reduced_chi2)
        self.add_metric("Correlation misfit", self._metric_corr_misfit)
        self.add_metric("Power misfit", self._metric_power_misfit)
        # add seismo-oriented metrics
        # self.add_metric("band_power_ratio", self._metric_band_power_ratio)
        self.add_metric("Envelope misfit", self._metric_envelope_misfit)
        self.add_metric("MSE", self._metric_mse)
        # self.add_metric("autocorr_misfit", self._metric_autocorr_misfit)

    # -------------------
    # Public API
    # -------------------
    def add_metric(self, name: str, func: Callable):
        """
        Register a new metric. func(obs, synthetics, meta) -> np.ndarray (len = n_synthetics)
        meta is a dict with helpful entries, e.g. sample_rate, n_traces, covariance_matrix, dof_override
        """
        if name in self.metrics:
            raise KeyError(f"Metric {name} already exists")
        self.metrics[name] = func
    def _metric_mse(self, obs, synthetics, meta):
        return np.mean((synthetics - obs) ** 2, axis=1)

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
        with tqdm_joblib(tqdm(desc="Simulating synthetics for PPCs", total=len(samples))) as progress_bar:
             with joblib.parallel_backend('loky', n_jobs=self.n_jobs):
                results = joblib.Parallel()( 
                     joblib.delayed(self.simulator)(param_dict) for param_dict in samples 
                )
                # Apply per-sample random time shifts if receivers are known
                if self.receivers is not None:
                    # derive distribution for this ensemble
                    if ensemble_name is not None and ensemble_name in self.random_shift_distributions:
                        print("Using random shift distribution for ensemble:", ensemble_name)
                        shifts = self.random_shift_distributions[ensemble_name]
                    receiver_names = [rec.station_name for rec in self.receivers.iterate()]
                    shifted = []
                    for vec in results:
                        try:
                            # Build per-station shift dict
                            if isinstance(shifts, tuple) and len(shifts) == 2:
                                random_shift_distribution= shifts
                                shift = round(np.random.normal(0, random_shift_distribution[0]))
                                shift_dict = {name: shift + int(np.random.uniform(-random_shift_distribution[1], random_shift_distribution[1])) for name in receiver_names}
                            # IMPORTANT: set shifts on receivers prior to applying
                            elif isinstance(shifts, dict):
                                shift_dict = shifts
                            else:
                                shift_dict = {name: 0 for name in receiver_names}
                            self.receivers.set_time_shifts(shift_dict)
                            # reshape vector -> outputs map, apply shifts, flatten back
                            outputs_map = self._vec_to_outputs_map(vec)
                            shifted_outputs = apply_station_time_shifts(self.receivers, outputs_map)
                            shifted_vec = self._outputs_map_to_vec(shifted_outputs)
                            shifted.append(shifted_vec)
                        except Exception:
                            shifted.append(vec)
                    synthetics.extend(shifted)
                else:
                    synthetics.extend(results) 
        return np.vstack(synthetics)

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
        meta = {
            "sample_rate": self.sample_rate,
            "n_traces": self.n_traces,
            "covariance_matrix": self.covariance_matrix,
            "dof_override": self.dof_override,
        }

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
        meta = {
            "sample_rate": self.sample_rate,
            "n_traces": self.n_traces,
            "covariance_matrix": self.covariance_matrix,
            "dof_override": self.dof_override,
        }

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

    # -----------------------
    # Built-in metric impls
    # Each returns 1D array of length n_synthetics
    # -----------------------

    def _chi2_from_cov(self, r: np.ndarray):
        """
        Try to compute r^T C^{-1} r using covariance_matrix interfaces;
        fall back to raw dot(r, r).
        """
        C = self.covariance_matrix
        if C is None:
            return float(np.dot(r, r))
        return -2* C.compute_loss(r, reduce=True)

    def _metric_reduced_chi2(self, obs: np.ndarray, synthetics: np.ndarray, meta: dict):
        # vectorized
        n_samples = synthetics.shape[0]
        obs_len = obs.size
        dof = meta.get("dof_override", None)
        if dof is None:
            dof = obs_len if obs_len > 0 else 1

        # compute residuals in vectorized fashion when possible
        diffs = synthetics - obs[None, :]  # shape (n_samples, obs_len)
        chi2s = np.empty((n_samples,), dtype=float)
        for i in range(n_samples):
            chi2s[i] = self._chi2_from_cov(diffs[i, :])
        # reduced
        return chi2s / float(dof)

    def _reshape_to_traces(self, vec: np.ndarray, n_traces: int):
        vec = np.asarray(vec).ravel()
        if n_traces is None or n_traces <= 0:
            return None
        if len(vec) % n_traces != 0:
            return None
        tlen = len(vec) // n_traces
        return vec.reshape((n_traces, tlen))

    def _metric_corr_misfit(self, obs: np.ndarray, synthetics: np.ndarray, meta: dict):
        """
        Per-sample misfit = mean over traces of (1 - Pearson corr) if trace shape known,
        else whole-vector (1 - corr) per sample.
        """
        n_samples = synthetics.shape[0]
        n_traces = meta.get("n_traces", self.n_traces)

        obs_mat = self._reshape_to_traces(obs, n_traces)
        if obs_mat is None:
            # fallback whole vector
            x = obs - obs.mean()
            xnorm = np.linalg.norm(x) + 1e-12
            vals = []
            for s in synthetics:
                y = s - s.mean()
                denom = xnorm * (np.linalg.norm(y) + 1e-12)
                corr = 0.0 if denom == 0.0 else float(np.dot(x, y) / denom)
                vals.append(1.0 - corr)
            return np.asarray(vals, dtype=float)

        # reshape synthetics -> (n_samples, n_traces, tlen)
        tlen = obs_mat.shape[1]
        try:
            syn_mat = synthetics.reshape((n_samples, obs_mat.shape[0], tlen))
        except Exception:
            # fallback
            return self._metric_corr_misfit(obs, synthetics, {**meta, "n_traces": None})

        x = obs_mat - obs_mat.mean(axis=1, keepdims=True)
        xnorm = np.linalg.norm(x, axis=1) + 1e-12  # [n_traces]
        vals = np.empty((n_samples,), dtype=float)
        for i in range(n_samples):
            y = syn_mat[i] - syn_mat[i].mean(axis=1, keepdims=True)
            ynorm = np.linalg.norm(y, axis=1) + 1e-12
            dots = np.sum(x * y, axis=1)
            corr_tr = np.where((xnorm * ynorm) == 0.0, 0.0, dots / (xnorm * ynorm))
            misfit_tr = 1.0 - corr_tr
            vals[i] = float(np.mean(misfit_tr))
        return vals

    def _metric_power_misfit(self, obs: np.ndarray, synthetics: np.ndarray, meta: dict):
        p_obs = float(np.dot(obs, obs)) + 1e-12
        p_syn = np.sum(synthetics ** 2, axis=1)
        vals = np.abs(p_obs - p_syn) / p_obs
        return vals

    def _metric_band_power_ratio(self, obs: np.ndarray, synthetics: np.ndarray, meta: dict, bands=None):
        """
        Compute band power misfit across frequency bands.
        Returns per-sample average relative L2-power difference across bands.

        bands: list of (fmin, fmax) in Hz. If None, default to
               low/mid/high: [(0.02, 0.2), (0.2, 1.0), (1.0, 10.0)] Hz — change to taste.
        Requires sample_rate in meta.
        """
        if welch is None:
            raise RuntimeError("scipy.signal.welch required for band_power_ratio metric")
        sr = meta.get("sample_rate", self.sample_rate)
        if sr is None:
            raise ValueError("sample_rate required for band_power_ratio metric")

        if bands is None:
            bands = [(0.05, 0.2), (0.2, 1.0),]
        # compute PSD for observation
        nperseg = max(256, int(sr))  # heuristic
        f_obs, P_obs = welch(obs, fs=sr, nperseg=nperseg)
        # helper to compute band power
        def band_power(f, P, bmin, bmax):
            mask = (f >= bmin) & (f < bmax)
            if not np.any(mask):
                return 0.0
            return float(np.trapz(P[mask], f[mask]))

        P_obs_bands = np.array([band_power(f_obs, P_obs, b0, b1) for (b0, b1) in bands]) + 1e-12

        # compute PSD for synthetics (vectorized loop)
        vals = np.empty((synthetics.shape[0],), dtype=float)
        for i, s in enumerate(synthetics):
            f_s, P_s = welch(s, fs=sr, nperseg=nperseg)
            # align by interpolating P_s onto f_obs if needed
            if not np.allclose(f_s, f_obs):
                P_s = np.interp(f_obs, f_s, P_s, left=0.0, right=0.0)
            P_s_bands = np.array([band_power(f_obs, P_s, b0, b1) for (b0, b1) in bands]) + 1e-12
            rel = np.abs(P_obs_bands - P_s_bands) / P_obs_bands
            vals[i] = float(np.mean(rel))
        return vals

    def _metric_envelope_misfit(self, obs: np.ndarray, synthetics: np.ndarray, meta: dict):
        """
        Envelope-based misfit: mean absolute relative difference of envelope over time.
        Requires scipy.signal.hilbert.
        """
        if hilbert is None:
            raise RuntimeError("scipy.signal.hilbert required for envelope_misfit metric")
        env_obs = np.abs(hilbert(obs))
        vals = np.empty((synthetics.shape[0],), dtype=float)
        for i, s in enumerate(synthetics):
            env_s = np.abs(hilbert(s))
            denom = env_obs + 1e-12
            vals[i] = float(np.mean(np.abs(env_obs - env_s) / denom))
        return vals

    def _metric_autocorr_misfit(self, obs: np.ndarray, synthetics: np.ndarray, meta: dict, maxlag=50):
        """
        Compare short-lag autocorrelation structure:
        statistic = mean_{lag=1..maxlag} |acf_obs(lag) - acf_syn(lag)|
        """
        obs = np.asarray(obs).ravel()
        n = len(obs)
        maxlag = min(maxlag, n - 1)
        # compute obs acf
        x = obs - obs.mean()
        denom = np.dot(x, x) + 1e-12
        acf_obs = np.array([np.dot(x[: n - lag], x[lag:]) / denom for lag in range(1, maxlag + 1)])
        vals = np.empty((synthetics.shape[0],), dtype=float)
        for i, s in enumerate(synthetics):
            y = s - s.mean()
            denom_y = np.dot(y, y) + 1e-12
            acf_s = np.array([np.dot(y[: n - lag], y[lag:]) / denom_y for lag in range(1, maxlag + 1)])
            vals[i] = float(np.mean(np.abs(acf_obs - acf_s)))
        return vals

    # -----------------------
    # Helpers for applying time shifts without code duplication
    # -----------------------
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
                    # support alt component names if needed
                    alt = comp.replace('E', '1').replace('N', '2')
                    trace = outputs[station].get(alt)
                if trace is None:
                    raise KeyError(f"Missing trace for {station}:{comp}")
                parts.append(np.asarray(trace))
        return np.concatenate(parts)

    # -----------------------
    # Helpers for selecting best-matching synthetics
    # -----------------------
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
            import traceback
            traceback.print_exc()
            return synthetics[:k], np.arange(k, dtype=int)
        # lower is better
        order = np.argsort(vals)
        best_idx = order[:k]
        return synthetics[best_idx], best_idx

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Optional, Union

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
        # norm_df[m] = norm_df[m].clip(upper=1.0)
    # norm_df = df
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
