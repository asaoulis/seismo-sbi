from typing import Iterable, List, Dict
import numpy as np
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import joblib
from scipy.stats import norm
import numpy as np
# Progress patch from your codebase
from seismo_sbi.instaseis_simulator.dataset_generator import tqdm_joblib
from tqdm import tqdm

# Use your existing utilities
from seismo_sbi.plotting.lune import mts6_to_gamma_delta, m6_to_matrix
from seismo_sbi.plotting.distributions import convert_to_pyrocko

# -----------------------------
# Helpers
# -----------------------------

def _wrap_angle_deg(d, period):
    h = period / 2.0
    return (d + h) % period - h

def _angle_diff_deg(a, b, period):
    return _wrap_angle_deg(a - b, period)

def _mw_from_m6(mt6_arr):
    M = m6_to_matrix(mt6_arr)               # (...,3,3)
    fro2 = np.sum(M * M, axis=(-2, -1))     # ||M||_F^2
    M0 = (1.0 / np.sqrt(2.0)) * np.sqrt(fro2)
    MW = (np.log10(np.clip(M0, 1e-30, np.inf)) - 9.1) / 1.5
    return MW

def _sdr_two_planes_from_mt(mt6):
    m = convert_to_pyrocko(np.asarray(mt6))
    sdrs = m.both_strike_dip_rake()  # [(strike1,dip1,rake1), (strike2,dip2,rake2)]
    return np.array(sdrs, dtype=float)

def _sdr_distance(a, b):
    ds = _angle_diff_deg(a[0], b[0], 360.0)
    dd = a[1] - b[1]
    dr = _angle_diff_deg(a[2], b[2], 360.0)
    return ds*ds + dd*dd + dr*dr

def _align_sdr_samples_to_truth_plane(theta0_mt, sample_mts):
    """
    Pick the truth nodal plane that minimises total distance to samples,
    then align each sample to the closer of its two planes.
    Returns: truth_sdr (3,), aligned_sdr_samples (n_samples, 3)
    """
    T = _sdr_two_planes_from_mt(theta0_mt)        # (2,3)
    A = []
    B = []
    for mt in sample_mts:
        sdr2 = _sdr_two_planes_from_mt(mt)
        A.append(sdr2[0]); B.append(sdr2[1])
    A = np.array(A)  # (n,3)
    B = np.array(B)  # (n,3)

    dsum_T0 = np.sum([min(_sdr_distance(A[i], T[0]), _sdr_distance(B[i], T[0])) for i in range(len(sample_mts))])
    dsum_T1 = np.sum([min(_sdr_distance(A[i], T[1]), _sdr_distance(B[i], T[1])) for i in range(len(sample_mts))])
    truth_idx = 0 if dsum_T0 <= dsum_T1 else 1
    truth_sdr = T[truth_idx]

    aligned = np.empty_like(A)
    for i in range(len(sample_mts)):
        dA = _sdr_distance(A[i], truth_sdr)
        dB = _sdr_distance(B[i], truth_sdr)
        aligned[i] = A[i] if dA <= dB else B[i]

    aligned[:, 0] = aligned[:, 0] % 360.0                   # strike
    aligned[:, 2] = _wrap_angle_deg(aligned[:, 2], 360.0)   # rake
    ts = truth_sdr.copy()
    ts[0] = ts[0] % 360.0
    ts[2] = _wrap_angle_deg(ts[2], 360.0)
    return ts, aligned

def _z_stats_linear(samples, truth):
    res = samples - truth
    mu = float(np.mean(res))
    sd = float(np.std(res, ddof=1))
    sd = sd if sd > 1e-12 else 1e-12
    z = mu / sd
    return mu, z

def _z_stats_circular(samples, truth, period):
    res = _angle_diff_deg(samples, truth, period)
    mu = float(np.mean(res))
    sd = float(np.std(res, ddof=1))
    sd = sd if sd > 1e-12 else 1e-12
    z = mu / sd
    return mu, z
def _circ_std_deg(samples, period):
    ang = np.asarray(samples) * 2*np.pi/period
    R = np.sqrt((np.mean(np.cos(ang)))**2 + (np.mean(np.sin(ang)))**2)
    # avoid negative due to FP
    R = np.clip(R, 0, 1)
    return (period/(2*np.pi)) * np.sqrt(-2*np.log(R))

# -----------------------------
# Compute API (per-parameter z)
# -----------------------------

def bias_z_from_posteriors_mt6_parallel(
    post_samples_mt6,
    truth_mt6,
    n_jobs=-1,
    max_samples_for_sdr=None
):
    """
    Inputs:
      post_samples_mt6: array (n_samples, n_events, 6)
      truth_mt6:        array (n_events, 6)
      n_jobs:           joblib parallel workers
      max_samples_for_sdr: optional subsample size for selecting nodal plane

    Returns:
      bias: dict of per-parameter arrays (n_events,)
      z:    dict of per-parameter arrays (n_events,)
      u:    dict with Phi(z) for compatibility (not plotted here)
    """
    post_samples_mt6 = np.asarray(post_samples_mt6)
    truth_mt6 = np.asarray(truth_mt6)
    assert post_samples_mt6.ndim == 3 and post_samples_mt6.shape[-1] == 6, "post_samples_mt6 must be (S,E,6)"
    assert truth_mt6.ndim == 2 and truth_mt6.shape[-1] == 6 and truth_mt6.shape[0] == post_samples_mt6.shape[1], \
        "truth_mt6 must be (E,6) with matching events"

    S, E, _ = post_samples_mt6.shape

    # Optional subsample for SDR truth-plane selection
    if max_samples_for_sdr is not None and S > max_samples_for_sdr:
        s_idx = np.random.permutation(S)[:max_samples_for_sdr]
        post_for_sdr = post_samples_mt6[s_idx]  # (S',E,6)
    else:
        post_for_sdr = post_samples_mt6

    # Vectorised gamma, delta, Mw across all samples and events
    flat = post_samples_mt6.reshape(-1, 6)            # (S*E,6)
    g_flat, d_flat = mts6_to_gamma_delta(flat)        # (S*E,), (S*E,)
    g_samp = g_flat.reshape(S, E)
    d_samp = d_flat.reshape(S, E)
    g0, d0 = mts6_to_gamma_delta(truth_mt6)           # (E,), (E,)

    mw_flat = _mw_from_m6(flat)                       # (S*E,)
    mw_samp = mw_flat.reshape(S, E)
    mw0 = _mw_from_m6(truth_mt6)                      # (E,)

    keys = ['gamma','delta','Mw','strike','dip','rake']

    def _event_worker(e):
        from scipy.special import erf
        out_bias = {}
        out_z    = {}
        out_u    = {}
        out_std  = {}          # <-- NEW

        # --- gamma ---
        mu, z = _z_stats_circular(g_samp[:, e], g0[e], period=60.0)
        out_bias['gamma'], out_z['gamma'] = mu, z
        out_u['gamma'] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        out_std['gamma'] = _circ_std_deg(g_samp[:, e], period=60.0)   # <-- NEW

        # --- delta ---
        mu, z = _z_stats_circular(d_samp[:, e], d0[e], period=180.0)
        out_bias['delta'], out_z['delta'] = mu, z
        out_u['delta'] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        out_std['delta'] = _circ_std_deg(d_samp[:, e], period=180.0)  # <-- NEW

        # --- Mw ---
        mu, z = _z_stats_linear(mw_samp[:, e], mw0[e])
        out_bias['Mw'], out_z['Mw'] = mu, z
        out_u['Mw'] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        out_std['Mw'] = np.nanstd(mw_samp[:, e], ddof=1)              # <-- NEW

        # -----------------------------
        # SDR selection (unchanged)
        # -----------------------------
        truth_sdr, _ = _align_sdr_samples_to_truth_plane(truth_mt6[e], post_for_sdr[:, e])

        strikes = np.empty(S)
        dips    = np.empty(S)
        rakes   = np.empty(S)
        for i in range(S):
            sdr2 = _sdr_two_planes_from_mt(post_samples_mt6[i, e])
            dA = _sdr_distance(sdr2[0], truth_sdr)
            dB = _sdr_distance(sdr2[1], truth_sdr)
            chosen = sdr2[0] if dA <= dB else sdr2[1]
            strikes[i] = chosen[0] % 360.0
            dips[i]    = chosen[1]
            rakes[i]   = _wrap_angle_deg(chosen[2], 360.0)

        s0 = truth_sdr[0] % 360.0
        d0p = truth_sdr[1]
        r0 = _wrap_angle_deg(truth_sdr[2], 360.0)

        # --- strike ---
        mu, z = _z_stats_circular(strikes, s0, period=360.0)
        out_bias['strike'], out_z['strike'] = mu, z
        out_u['strike'] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        out_std['strike'] = _circ_std_deg(strikes, period=360.0)       # <-- NEW

        # --- dip ---
        mu, z = _z_stats_linear(dips, d0p)
        out_bias['dip'], out_z['dip'] = mu, z
        out_u['dip'] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        out_std['dip'] = np.nanstd(dips, ddof=1)                       # <-- NEW

        # --- rake ---
        mu, z = _z_stats_circular(rakes, r0, period=360.0)
        out_bias['rake'], out_z['rake'] = mu, z
        out_u['rake'] = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
        out_std['rake'] = _circ_std_deg(rakes, period=360.0)           # <-- NEW

        return out_bias, out_z, out_u, out_std


    with tqdm_joblib(tqdm(desc="Computing events", total=E)) as _pbar:
        results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
            joblib.delayed(_event_worker)(e) for e in range(E)
        )

    # Unpack lists
    bias_list, z_list, u_list, std_list = zip(*results)   # <-- NEW

    bias = {k: np.array([b[k] for b in bias_list]) for k in keys}
    z    = {k: np.array([b[k] for b in z_list])   for k in keys}
    u    = {k: np.array([b[k] for b in u_list])   for k in keys}
    std  = {k: np.array([b[k] for b in std_list]) for k in keys}   # <-- NEW

    return bias, z, u, std
# -----------------------------
# Plotting multiple ensembles
# -----------------------------
def extract_experiment_results_by_keys(
    folders: Iterable[str],
    keys: List[str],
    base_folder: str,
):
    """
    Collect inversion results filtered by substrings in `inversion_method`.

    Parameters
    ----------
    folders : iterable of str
        Folder names (relative to base_folder) that contain 'inversion_results.pkl'.
    keys : list of str
        Substrings to look for inside InversionConfig.inversion_method.
        For each key, we gather all results whose inversion_method contains that key.
    base_folder : str
        Base directory where job folders live.
    param_names : dict
        Mapping param_type -> list of param_names defining the order
        in which to extract theta0 from the theta0_dict.

    Returns
    -------
    results_by_key : dict
        {
          key: {
            "results":      list of np.ndarray  # scaled samples for that key
            "raw_results":  list of np.ndarray  # unscaled samples for that key
            "theta0s":      list of np.ndarray  # scaled theta0 vectors
            "raw_theta0s":  list of np.ndarray  # unscaled theta0 vectors
          },
          ...
        }
    x0s : np.ndarray
        Array of x0 (compressed_job) across all processed jobs.
    theta0_dicts : list
        List of theta0_dicts (one per job entry).
    """

    # Prepare container per key
    results_by_key = {
        key: {
            "results": [],
            "raw_results": [],
            "theta0s": [],
            "raw_theta0s": [],
        }
        for key in keys
    }

    x0s = []
    theta0_dicts = []

    for folder in folders:
        pkl_path = os.path.join(base_folder, folder, "inversion_results.pkl")
        with open(pkl_path, "rb") as f:
            job_data_list, inversion_data_list, inversion_results_list = pickle.load(f)

        # Sanity: keep them as lists so we can index
        job_data_list = list(job_data_list)
        inversion_data_list = list(inversion_data_list)
        inversion_results_list = list(inversion_results_list)

        for inv_res in inversion_results_list:


            inv_data = inv_res.inversion_data
            inv_cfg = inv_res.inversion_config

            data_scaler = inv_data.data_scaler
            samples = inv_data.samples  # shape: (n_samples, dim)

            # Find which keys this inversion_method matches
            matching_keys = [k for k in keys if k in inv_cfg.inversion_method]
            # print(f"Found matching keys for inversion_method '{inv_cfg.inversion_method}': {matching_keys}")
            # if multiple keys find exact match
            if len(matching_keys) > 1:
                matching_keys = [k for k in matching_keys if k == inv_cfg.inversion_method]
            if not matching_keys:
                continue

            if inv_data.theta0 is not None:
                raw_theta0 = np.array(inv_data.theta0, dtype=float).ravel()
            else:

                raise RuntimeError(
                    "InversionData.theta0 is None; please provide "
                    "a mapping from InversionResult to job_data to build theta0."
                )

            # Scale theta0 and samples
            theta0_scaled = data_scaler.transform(raw_theta0.reshape(1, -1)).reshape(-1)
            samples_scaled = data_scaler.transform(samples)

            # Unscaled samples for convenience
            samples_unscaled = data_scaler.inverse_transform(samples_scaled)

            # Save x0 if available from compression_data or similar
            if hasattr(inv_data, "compression_data") and getattr(inv_data.compression_data, "compressed_job", None) is not None:
                x0 = inv_data.compression_data.compressed_job
                x0s.append(x0)

            # Store to all matching keys
            for key in matching_keys:
                results_by_key[key]["results"].append(samples_scaled)
                results_by_key[key]["raw_results"].append(samples_unscaled)
                results_by_key[key]["theta0s"].append(theta0_scaled)
                results_by_key[key]["raw_theta0s"].append(raw_theta0)

        
            # Optionally keep raw theta0_dicts if available in JobData or InversionData
            if hasattr(inv_data, "theta0_dict"):
                theta0_dicts.append(inv_data.theta0_dict)
    # convert all lists to np.ndarrays
    for key in keys:
        for subkey in results_by_key[key]:
            results_by_key[key][subkey] = np.array(results_by_key[key][subkey])

    return results_by_key, np.array(x0s), theta0_dicts