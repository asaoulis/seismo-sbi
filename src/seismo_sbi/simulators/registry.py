"""Build the forward model a configuration's ``simulation_type`` asks for.

One builder per name in :data:`SIMULATOR_REGISTRY`; :func:`build_simulator` looks the name up
and calls it with the simulation parameters. A forward model that lives outside this repository
joins by calling :func:`register_simulator` with its own builder, after which its name is a
valid ``simulation_type``.
"""

import json
from copy import deepcopy
from pathlib import Path

from .receivers import Receivers
from .kernel import FixedLocationKernelSimulator
from .theory_covariance import EnsembleTheoryCovarianceEstimationSimulator
from .instaseis.simulator import InstaseisSourceSimulator
from .instaseis.ensemble import InstaseisEnsembleSimulator
from .instaseis.multi_model import InstaseisMultiModelSimulator
from .cps.simulator import (
    CPSVariableKernelSimulator, CPSPrecomputedSimulator, MultiModelCPSSimulator,
)


def _sub_receivers(station_to_receiver, station_names, where):
    """The receivers of ``station_names``, looked up in ``station_to_receiver``; ``where`` names the
    configuration entry in the error for a station that is not among the receivers."""
    sub_receivers_list = []
    for sta in station_names:
        try:
            sub_receivers_list.append(station_to_receiver[sta])
        except KeyError:
            raise KeyError(f"Station '{sta}' in {where} not found in global receivers list")
    return Receivers(receivers=sub_receivers_list)


def _build_cps_multi_models_from_path(simulation_parameters):
    """Sub-model dicts read from the JSON file ``cps_multi_models_path`` names.

    The file holds a list of objects, each with ``cps_GFs_path``, ``cps_GFs_fiducial_path``
    and ``receivers``, a list of station names.
    """
    cfg_path_str = simulation_parameters.cps_multi_models_path
    if not cfg_path_str:
        return None

    cfg_path = Path(cfg_path_str)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_list = json.load(f)

    if not isinstance(cfg_list, list):
        raise ValueError(
            f"cps_multi_models_path JSON must contain a list of model configs, got {type(cfg_list)}"
        )

    global_receivers = simulation_parameters.receivers
    station_to_receiver = {rec.station_name: rec for rec in global_receivers.iterate()}

    models = []
    for idx, cfg in enumerate(cfg_list):
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Entry {idx} in {cfg_path} must be an object/dict, got {type(cfg)}"
            )

        sub_receivers = _sub_receivers(station_to_receiver, cfg.get("receivers", []), f"{cfg_path} (entry {idx})")
        model_cfg = {
            "receivers": sub_receivers,
            "cps_GFs_path": cfg["cps_GFs_path"],
            "cps_GFs_fiducial_path": cfg["cps_GFs_fiducial_path"],
        }
        models.append(model_cfg)
    return models


def _build_instaseis_multi_models(simulation_parameters):
    """Turn the inline ``instaseis_multi_models`` YAML list into sub-model dicts.

    Each YAML entry has ``ensemble_dir`` + ``fiducial_dir`` (Instaseis DB ensemble paths) and
    ``receivers`` (a list of station names). The database paths are kept inline as YAML string
    leaves so a cluster orchestrator can rewrite them to its own archive.
    """
    cfg_list = simulation_parameters.instaseis_multi_models
    if not cfg_list:
        return None

    if not isinstance(cfg_list, list):
        raise ValueError(
            f"instaseis_multi_models must be a list of model configs, got {type(cfg_list)}"
        )

    global_receivers = simulation_parameters.receivers
    station_to_receiver = {rec.station_name: rec for rec in global_receivers.iterate()}

    models = []
    for idx, cfg in enumerate(cfg_list):
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Entry {idx} in instaseis_multi_models must be an object/dict, got {type(cfg)}"
            )

        sub_receivers = _sub_receivers(station_to_receiver, cfg.get("receivers", []),
                                       f"instaseis_multi_models (entry {idx})")
        models.append({
            "receivers": sub_receivers,
            "ensemble_dir": cfg["ensemble_dir"],
            "fiducial_dir": cfg["fiducial_dir"],
        })
    return models


def _build_instaseis_ensemble(simulation_parameters, simulator_config, pp_effects, data_flattening):
    return InstaseisEnsembleSimulator(
                    instaseis_ensemble_dir=simulation_parameters.syngine_address,
                    instaseis_fiducial_loc=simulation_parameters.syngine_fiducial_address,
                    components=simulation_parameters.components,
                    receivers=simulation_parameters.receivers,
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment,
                    resample_member_per_station=simulation_parameters.resample_member_per_station,
                    member_sampling=simulation_parameters.member_sampling,
                    sector_lambda=simulation_parameters.sector_lambda,
                    source_depth_offset_km=simulation_parameters.source_depth_offset_km)


def _build_instaseis(simulation_parameters, simulator_config, pp_effects, data_flattening):
    return InstaseisSourceSimulator(simulation_parameters.syngine_address,
                                components=simulation_parameters.components,
                                receivers=simulation_parameters.receivers,
                                seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                                synthetics_processing=simulation_parameters.processing,
                                post_processing_effects=pp_effects,
                                stf_alignment=simulation_parameters.stf_alignment,
                                source_depth_offset_km=simulation_parameters.source_depth_offset_km)


def _build_kernel(simulation_parameters, simulator_config, pp_effects, data_flattening):
    score_compression_data = simulator_config[1]
    return FixedLocationKernelSimulator(score_compression_data,
                    components=simulation_parameters.components,
                    receivers=simulation_parameters.receivers,
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment)


def _build_cps(simulation_parameters, simulator_config, pp_effects, data_flattening):
    return CPSVariableKernelSimulator(
                    components=simulation_parameters.components,
                    receivers=simulation_parameters.receivers,
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    gf_storage_root=simulation_parameters.cps_GFs_path,
                    cps_path=simulation_parameters.cps_path,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment)


def _build_cps_precomputed(simulation_parameters, simulator_config, pp_effects, data_flattening):
    return CPSPrecomputedSimulator(
                    fiducial_model_path=simulation_parameters.cps_GFs_fiducial_path,
                    components=simulation_parameters.components,
                    receivers=simulation_parameters.receivers,
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    gf_storage_root=simulation_parameters.cps_GFs_path,
                    cps_path=simulation_parameters.cps_path,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment)


def _build_instaseis_multi_ensemble(simulation_parameters, simulator_config, pp_effects, data_flattening):
    if simulator_config[1] is not None:
        models = simulator_config[1]
    else:
        models = _build_instaseis_multi_models(simulation_parameters)
    if not models:
        raise ValueError(
            "simulation_type 'instaseis_multi_ensemble' requires a non-empty "
            "'instaseis_multi_models' list in seismic_context."
        )
    return InstaseisMultiModelSimulator(
                    models=models,
                    components=simulation_parameters.components,
                    receivers=simulation_parameters.receivers,
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment,
                    resample_member_per_station=simulation_parameters.resample_member_per_station,
                    member_sampling=simulation_parameters.member_sampling,
                    sector_lambda=simulation_parameters.sector_lambda,
                    source_depth_offset_km=simulation_parameters.source_depth_offset_km)


def _build_cps_multi(simulation_parameters, simulator_config, pp_effects, data_flattening):
    if simulator_config[1] is not None:
        models = simulator_config[1]
    else:
        models = _build_cps_multi_models_from_path(simulation_parameters)
    if not models:
        raise ValueError(
            "simulation_type 'cps_multi' requires either explicit models "
            "or a non-empty cps_multi_models_path in SimulationParameters."
        )
    return MultiModelCPSSimulator(
                    models=models,
                    components=simulation_parameters.components,
                    receivers=simulation_parameters.receivers,
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    cps_path=simulation_parameters.cps_path,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment)


def _build_theory_covariance(simulation_parameters, simulator_config, pp_effects, data_flattening):
    ensemble_simulator = simulator_config[1]
    return EnsembleTheoryCovarianceEstimationSimulator(
                    simulator=ensemble_simulator,
                    data_flattening=data_flattening,
                    components=simulation_parameters.components,
                    receivers=deepcopy(simulation_parameters.receivers),
                    seismogram_duration_in_s=simulation_parameters.seismogram_duration,
                    synthetics_processing=simulation_parameters.processing,
                    post_processing_effects=pp_effects,
                    stf_alignment=simulation_parameters.stf_alignment)


#: Forward-model builders, selectable by ``simulation_type``. Each takes the simulation
#: parameters, the ``(name, payload)`` configuration tuple, the post-processing effects and the
#: callable that flattens a simulation into a data vector, and returns a ``Simulator``.
SIMULATOR_REGISTRY = {
    "instaseis": _build_instaseis,
    "instaseis_ensemble": _build_instaseis_ensemble,
    "instaseis_multi_ensemble": _build_instaseis_multi_ensemble,
    "kernel": _build_kernel,
    "cps": _build_cps,
    "cps_precomputed": _build_cps_precomputed,
    "cps_multi": _build_cps_multi,
    "theory_covariance": _build_theory_covariance,
}


def register_simulator(name: str, builder) -> None:
    """Make ``name`` a valid ``simulation_type`` served by ``builder``."""
    SIMULATOR_REGISTRY[name] = builder


def build_simulator(simulator_config, simulation_parameters, post_processing_effects=None,
                    data_flattening=None):
    """Build the simulator ``simulator_config[0]`` names.

    ``simulator_config`` is ``(simulation_type, payload)``; the payload is the kernel data, the
    ensemble simulator or an explicit sub-model list, depending on the type.
    """
    name = simulator_config[0]
    builder = SIMULATOR_REGISTRY.get(name)
    if builder is None:
        raise NotImplementedError(
            f"Simulator {name} not implemented; known types are {sorted(SIMULATOR_REGISTRY)}"
        )
    return builder(simulation_parameters, simulator_config, post_processing_effects or [],
                   data_flattening)
