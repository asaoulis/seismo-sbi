import os
import argparse
import pickle
# prevent processes from using multiple threads
# this is necessary because otherwise the multiprocessing
# in emcee may use more threads than requested
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
# removed top-level h5py import to avoid missing dependency error during static checks
import numpy as np

from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.sbi.noises.covariance_estimation import (
    BlockDiagonalFilteredCovariance,
    BlockDiagonalEmpiricalCovariance,
    DiagonalEmpiricalCovariance,
    BlockDiagonalKolbCovariance,
)


class NoiseComparisonConfig:
    """
    Holds configuration parameters for the noise covariance comparison.
    """
    def __init__(
        self,
        config_path: Path,
        noise_glob: str,
        freqmin: float,
        freqmax: float,
        reload_empirical: bool = False,
    ):
        self.config_path = config_path
        self.noise_glob = noise_glob
        self.freqmin = freqmin
        self.freqmax = freqmax
        self.reload_empirical = reload_empirical


def init_pipeline(config_path: Path):
    config = SBI_Configuration()
    config.parse_config_file(config_path)
    sbi_pipeline = SingleEventPipeline(config.pipeline_parameters, config_path)
    sbi_pipeline.load_seismo_parameters(
        config.sim_parameters, config.model_parameters, config.dataset_parameters
    )
    return sbi_pipeline, config


def list_noise_files(noise_glob: str) -> List[Path]:
    from glob import glob
    return [Path(p) for p in sorted(glob(noise_glob))]


def load_traces_dict(noise_file: Path) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Load per station-component traces from an HDF5 noise file.
    Returns dict keyed by (station, component) -> ndarray(time_series).
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        raise RuntimeError("h5py is required to read noise files. Please install it (e.g., pip install h5py).")
    traces = {}
    with h5py.File(noise_file, "r") as f:
        outputs = f["outputs"]
        for station in outputs.keys():
            grp = outputs[station]
            for component in grp.keys():
                try:
                    data = grp[component][:]
                except Exception:
                    continue
                traces[(station, component)] = data
    return traces


def load_misc_covariances(sbi_pipeline, noise_file: Path):
    """
    Load misc data containing pre-window auto-covariance per station-component.
    Expected to be a dict: station -> component -> covariance array.
    """
    misc = sbi_pipeline.data_manager.load_noise_parametrisation_data(noise_file)
    return misc


def build_diag_covariances_from_lag0(station_component_covariances: Dict, data_len: int) -> Dict:
    """
    Build covariances where only lag-0 is kept, yielding diagonal covariance blocks.
    For each station-component, create a covariance vector [sigma2, 0, 0, ...] of length data_len.
    """
    diag_covs = {}
    for station, comp_dict in station_component_covariances.items():
        diag_covs[station] = {}
        for component, cov_vec in comp_dict.items():
            # Use first element as variance (lag 0)
            if hasattr(cov_vec, "__len__") and len(cov_vec) > 0:
                sigma2 = float(cov_vec[0])
            else:
                sigma2 = float(cov_vec)
            diag_covs[station][component] = np.array(sigma2)
    return diag_covs

def build_filtered_cov_sigma2_dict(station_component_covariances: Dict) -> Dict:
    """
    For filtered covariance, BlockDiagonalFilteredCovariance expects either a dict with sigma^2
    per station-component or a scalar. We pass the lag-0 variance.
    """
    sigma2_dict = {}
    for station, comp_dict in station_component_covariances.items():
        sigma2_dict[station] = {}
        for component, cov_vec in comp_dict.items():
            if hasattr(cov_vec, "__len__") and len(cov_vec) > 0:
                sigma2 = float(cov_vec[0])
            else:
                sigma2 = float(cov_vec)
            sigma2_dict[station][component] = sigma2
    return sigma2_dict

def build_exp_cov_sigma2_dict(station_component_covariances: Dict, data_len) -> Dict:
    """
    For exp-tapered covariance, BlockDiagonalEmpiricalCovariance wants a standardised dict of vectors;
    just trim the existiting covariances to sigma^2 for each station-component.
    """
    sigma2_dict = {}
    for station, comp_dict in station_component_covariances.items():
        sigma2_dict[station] = {}
        for component, cov_vec in comp_dict.items():
            sigma2_dict[station][component] = np.ones(data_len) * float(cov_vec)
    return sigma2_dict


def build_covariance_objects(receivers, components: str, data_len: int, station_component_covariances: Dict, freqmin: float, freqmax: float):
    """
    Build four covariance objects: diagonal (lag-0), exp-tapered empirical, filtered bandpass, and Kolb.
    """
    # Diagonal via BlockDiagonalEmpiricalCovariance with toeplitz([sigma2, 0, ...]) -> diagonal matrix
    diag_covs = build_diag_covariances_from_lag0(station_component_covariances, data_len)
    cov_diag = DiagonalEmpiricalCovariance(
        station_component_covariances=diag_covs,
        receivers=receivers,
        data_vector_length=data_len,
    )

    standardised_station_component_covariances = build_exp_cov_sigma2_dict(station_component_covariances, data_len)
    cov_taper = BlockDiagonalEmpiricalCovariance(
        station_component_covariances=standardised_station_component_covariances,
        receivers=receivers,
        data_vector_length=data_len,
        block_exp_tapering=True,
        covariance_gradients=None,
        num_jobs=10,
    )

    # Filtered bandpass using sigma^2 and freqs
    sigma2_dict = build_filtered_cov_sigma2_dict(station_component_covariances)
    cov_filt = BlockDiagonalFilteredCovariance(
        station_component_covariances=sigma2_dict,
        filter={"freqmin": freqmin, "freqmax": freqmax},
        receivers=receivers,
        data_vector_length=data_len,
        block_exp_tapering=True,
        covariance_gradients=None,
        num_jobs=10,
    )

    cov_kolb = BlockDiagonalKolbCovariance(
        station_component_covariances=sigma2_dict,
        receivers=receivers,
        data_vector_length=data_len,
        omega_0=4.4,
        lam=0.2,
        block_exp_tapering=True,
        covariance_gradients=None,
        num_jobs=10,
    )

    return cov_diag, cov_taper, cov_filt, cov_kolb


def compute_reduced_chi2_per_block_via_loss(cov_obj, traces: Dict[Tuple[str, str], np.ndarray], receivers, data_len: int) -> Dict[Tuple[str, str], float]:
    """
    Use the covariance class generic_loss_callable(reduce=False) to compute per-block losses.
    Convert to chi2 per block: chi2 = -2 * loss_block, then reduce by dof=data_len.
    """
    # Build residual vector r in the same ordering as receivers/components
    residuals_list = []
    keys_list = []
    for receiver in receivers.iterate():
        for component in receiver.components:
            key = (receiver.station_name, component)
            alt_key = (receiver.station_name, component.replace("E", "1").replace("N", "2"))
            vec = traces.get(key) if key in traces else traces.get(alt_key)
            if vec is None:
                # pad zeros if missing to maintain alignment
                residuals_list.append(np.zeros(data_len))
                keys_list.append(key)
                continue
            residuals_list.append(np.asarray(vec[:data_len]))
            keys_list.append(key)
    residuals = np.concatenate(residuals_list)

    # Compute elementwise repeated block losses
    losses_elementwise = cov_obj.generic_loss_callable(residuals, reduce=False)
    # Recover per-block loss by taking the first element of each block (all entries in a block are equal)
    losses_blocks = losses_elementwise.reshape(-1, data_len)[:, 0]

    chi2s = {}
    for key, loss in zip(keys_list, losses_blocks):
        chi2 = -2.0 * float(loss)
        red = chi2 / data_len
        chi2s[key] = red
    return chi2s


def process_noise_file(noise_file: Path, sbi_pipeline, freqmin: float, freqmax: float):
    traces = load_traces_dict(noise_file)
    misc_covs = load_misc_covariances(sbi_pipeline, noise_file)
    receivers = sbi_pipeline.data_manager.data_loader.receivers
    components = sbi_pipeline.data_manager.data_loader.components

    # Determine data length from any trace
    any_vec = next(iter(traces.values()))
    data_len = len(any_vec)

    cov_diag, cov_taper, cov_filt, cov_kolb = build_covariance_objects(
        receivers, components, data_len, misc_covs, freqmin, freqmax
    )

    # Compute per-block reduced chi^2 via loss
    chi2_diag = compute_reduced_chi2_per_block_via_loss(cov_diag, traces, receivers, data_len)
    chi2_taper = compute_reduced_chi2_per_block_via_loss(cov_taper, traces, receivers, data_len)
    chi2_filt = compute_reduced_chi2_per_block_via_loss(cov_filt, traces, receivers, data_len)
    chi2_kolb = compute_reduced_chi2_per_block_via_loss(cov_kolb, traces, receivers, data_len)

    return {
        "diag": chi2_diag,
        "taper": chi2_taper,
        "filtered": chi2_filt,
        "kolb": chi2_kolb,
    }

from tqdm import tqdm
def aggregate_results(noise_files: List[Path], sbi_pipeline, freqmin: float, freqmax: float):
    """
    Process all files and aggregate reduced chi^2 lists per station-component per covariance type.
    Returns dict: cov_type -> {(station, component): [red_chi2_values...]}
    """
    aggregated = {"diag": {}, "taper": {}, "filtered": {}, "kolb": {}}
    for nf in tqdm(noise_files, desc="Processing noise files"):
        res = process_noise_file(nf, sbi_pipeline, freqmin, freqmax)
        for cov_type, chi2_map in res.items():
            for key, val in chi2_map.items():
                aggregated[cov_type].setdefault(key, []).append(val)
                # print cov type and chi2 map
                # print(f"Covariance type: {cov_type}, Station-Component: {key}, Reduced Chi^2: {val}")

    return aggregated


def aggregate_by_covariance_and_component(aggregated: Dict[str, Dict[Tuple[str, str], List[float]]]):
    """
    Build a pandas DataFrame summarizing reduced chi^2 by covariance type and component.
    Columns: [covariance_type, component, count, mean, std]
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        raise RuntimeError("pandas is required to print the summary DataFrame. Please install it (e.g., pip install pandas).")

    rows = []
    for cov_type, comp_map in aggregated.items():
        # group by component only, across stations
        component_groups: Dict[str, List[float]] = {}
        for (station, component), values in comp_map.items():
            component_groups.setdefault(component, []).extend(values)
        for component, vals in component_groups.items():
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "covariance_type": cov_type,
                "component": component,
                "count": int(arr.size),
                "mean": float(arr.mean()) if arr.size else np.nan,
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            })
    df = pd.DataFrame(rows)
    # order columns
    df = df[["covariance_type", "component", "count", "mean", "std"]]
    print(df)
    return df

from typing import Dict, Tuple, List

def plot_violin_reduced_chi2(
    aggregated: Dict[str, Dict[Tuple[str, str], List[float]]],
    component: str,
    title: str = None,
):
    """
    Create violin plots of reduced chi^2 distributions for a given component across methods.
    Uses seaborn with native log scaling.
    """
    try:
        import seaborn as sns  # type: ignore
        import pandas as pd    # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "seaborn, pandas, and matplotlib are required for plotting "
            "(pip install seaborn pandas matplotlib)."
        ) from e

    methods = ["diag", "taper", "filtered", "kolb"]
    method_labels = {
        "diag": "Diagonal",
        "taper": "Exp-Tapered",
        "filtered": "Filtered",
        "kolb": "Kolb",
    }

    # support alternate component labels (E/N)
    alt = component.replace("1", "E").replace("2", "N")

    # ---- build tidy dataframe ----
    records = []
    for m in methods:
        comp_map = aggregated.get(m, {})
        for (station, comp), values in comp_map.items():
            if comp == component or comp == alt:
                for v in values:
                    if v > 0:  # log-scale safety
                        records.append(
                            {
                                "Method": method_labels[m],
                                "ReducedChi2": v,
                            }
                        )

    if not records:
        raise ValueError(f"No data found for component {component}")

    df = pd.DataFrame.from_records(records)

    # ---- plotting ----
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(
        data=df,
        x="Method",
        y="ReducedChi2",
        log_scale=True,        # ← seaborn-native log scaling
        inner="quartile",      # shows median + IQR
        cut=0,
        linewidth=1,
        ax=ax,
        color="#1f77b4",
    )

    # reference line at chi^2 = 1
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    ax.set_ylabel(r"Reduced $\chi^2$")
    if title is None:
        title = rf"Reduced $\chi^2$ distributions for component {component}"
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


