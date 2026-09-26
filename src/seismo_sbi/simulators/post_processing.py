"""The post-processing chain: effects applied in order to synthetic seismograms.

``PostProcessingChain`` composes ``SeismogramEffect`` objects, each one's output feeding the
next. ``build_post_processing_chain(nuisance_keys)`` assembles a chain from ``EFFECT_REGISTRY``,
silently skipping keys that name no effect, so a caller can pass every nuisance key it has.
The augmentation builders and ``apply_chain_to_array`` run the same effects in the dataloader.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from seismo_sbi.simulators.amplitude_effect import AmplitudeErrorEffect
from seismo_sbi.simulators.anisotropy_effects import AzimuthalAnisotropyEffect, ShearSplittingEffect
from seismo_sbi.simulators.dispersion_effect import DispersionSpreadEffect
from seismo_sbi.simulators.dropout_effects import ComponentDropoutEffect, InstrumentDropoutEffect
from seismo_sbi.simulators.scattering_coda_effect import ScatteringCodaEffect
from seismo_sbi.simulators.seismogram_effect import SeismogramEffect
from seismo_sbi.simulators.time_shift_effect import TimeShiftErrorEffect


class PostProcessingChain:
    """Effects applied in order, each one's output feeding the next; an empty chain is
    the identity.
    """

    def __init__(self, effects: list[SeismogramEffect] | None = None) -> None:
        self.effects: list[SeismogramEffect] = list(effects or [])

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        nuisance_params: dict,
    ) -> dict:
        """The seismogram map with every effect applied in order."""
        result = seismograms_map
        for effect in self.effects:
            result = effect(result, receivers, **nuisance_params)
        return result


#: Nuisance parameter key to effect class; a new effect is registered here.
EFFECT_REGISTRY: dict[str, type[SeismogramEffect]] = {
    "amplitude_error": AmplitudeErrorEffect,
    "instrument_dropout": InstrumentDropoutEffect,
    "time_shift_error": TimeShiftErrorEffect,
    "scattering_coda": ScatteringCodaEffect,
    "component_dropout": ComponentDropoutEffect,
    "azimuthal_anisotropy": AzimuthalAnisotropyEffect,
    "shear_wave_splitting": ShearSplittingEffect,
    "dispersion_spread": DispersionSpreadEffect,
}


#: Nuisance keys eligible for pre-noise training augmentation, folded into the clean signal.
AUGMENTABLE_EFFECT_KEYS: tuple[str, ...] = (
    "amplitude_error",
    "instrument_dropout",
    "time_shift_error",
    "scattering_coda",
)


#: Nuisance keys eligible for post-noise augmentation, applied to the data plus noise.
#: ``component_dropout`` must run there so a dropped channel is exactly zero.
POST_NOISE_EFFECT_KEYS: tuple[str, ...] = (
    "component_dropout",
)


#: Nuisance keys that augment the source-location conditioning vector rather than the
#: waveform, so they have no effect class and the dataloader applies them.
CONDITIONING_AUGMENTABLE_KEYS: tuple[str, ...] = (
    "source_location_error",
)


#: The effect keys eligible at each augmentation stage.
_STAGE_EFFECT_KEYS: dict[str, tuple[str, ...]] = {
    "training_augmentation": AUGMENTABLE_EFFECT_KEYS,
    "training_augmentation_post_noise": POST_NOISE_EFFECT_KEYS,
}


# Map <-> stacked-array adapter (lets the SAME effects run in the dataloader)


def _array_to_map(D: np.ndarray, receivers, components):
    """``(seismograms map, station names)`` from a ``(n_stations, n_components, n_samples)``
    array, in receiver order and loader component order.
    """
    station_names = [rec.station_name for rec in receivers.iterate()]
    seismograms_map = {
        station: {comp: D[i, j] for j, comp in enumerate(components)}
        for i, station in enumerate(station_names)
    }
    return seismograms_map, station_names


def _map_to_array(seismograms_map: dict, station_names, components) -> np.ndarray:
    """The inverse of :func:`_array_to_map`: back to ``(n_stations, n_components, n_samples)``."""
    return np.array(
        [[seismograms_map[station][comp] for comp in components] for station in station_names],
        dtype=np.float64,
    )


def apply_chain_to_array(
    chain: PostProcessingChain,
    D: np.ndarray,
    receivers,
    components,
    nuisance_params: dict,
) -> np.ndarray:
    """``D``, shaped ``(n_stations, n_components, n_samples)``, with ``chain`` applied.

    Lets the same effect classes serve as training-time augmentation on the dataloader's
    stacked array. ``receivers`` and ``components`` give the station and component orders that
    array is in. An empty chain returns ``D`` unchanged.
    """
    if not chain.effects:
        return D
    D = np.asarray(D)
    # A silent mismatch would scramble stations rather than raise.
    n_stations = len(list(receivers.iterate()))
    if D.ndim != 3 or D.shape[0] != n_stations or D.shape[1] != len(components):
        raise ValueError(
            f"apply_chain_to_array: D shape {D.shape} is incompatible with "
            f"{n_stations} receivers x {len(components)} components "
            f"(expected ({n_stations}, {len(components)}, T))."
        )
    seismograms_map, station_names = _array_to_map(D, receivers, components)
    processed = chain(seismograms_map, receivers, nuisance_params)
    return _map_to_array(processed, station_names, components)


def _fiducial_scalar(value) -> float:
    """The activation scalar of a nuisance fiducial entry, so ``[0.3]`` gives 0.3."""
    arr = np.ravel(value)
    return float(arr[0])


def build_augmentation_chain(
    nuisance: dict,
    nuisance_stage: dict,
    effect_configs: Optional[dict] = None,
    sampling_rate: Optional[float] = None,
    stage: str = "training_augmentation",
) -> Tuple[PostProcessingChain, dict]:
    """``(chain, nuisance_params)`` for the training-time augmentation at ``stage``.

    ``nuisance`` is ``{key: fiducial values}`` and ``nuisance_stage`` is ``{key: stage}``, a
    key absent from it defaulting to the simulation stage and so not augmented. Only keys
    staged at ``stage`` and eligible there are included. Each key's activation value in
    ``nuisance_params`` is its configured fiducial scalar; the magnitudes live in
    ``effect_configs``, and the effects draw their own randomness per call. ``sampling_rate``
    in samples per second is injected into the shift effect. An empty selection gives an empty
    chain, which the dataloader treats as no augmentation.
    """
    configs = dict(effect_configs or {})
    eligible = _STAGE_EFFECT_KEYS.get(stage, ())
    aug_keys = [
        key for key in nuisance
        if key in eligible
        and nuisance_stage.get(key, "simulation") == stage
    ]
    if "time_shift_error" in aug_keys and sampling_rate is not None:
        configs["time_shift_error"] = dict(configs.get("time_shift_error", {}))
        configs["time_shift_error"]["sampling_rate"] = sampling_rate

    chain = build_post_processing_chain(aug_keys, configs)
    # The configured fiducial value, not a hardcoded 1.0, which would perturb every station.
    nuisance_params = {key: _fiducial_scalar(nuisance[key]) for key in aug_keys}
    return chain, nuisance_params


def build_augmentation_chain_from_parameters(parameters, sampling_rate=None,
                                             stage="training_augmentation"):
    """``(chain, nuisance_params)`` as :func:`build_augmentation_chain`, unpacking the
    nuisance blocks from a parsed ``ModelParameters``.
    """
    return build_augmentation_chain(
        parameters.nuisance,
        getattr(parameters, "nuisance_stage", {}),
        getattr(parameters, "nuisance_effect_config", {}),
        sampling_rate=sampling_rate,
        stage=stage,
    )


def build_post_processing_chain(
    nuisance_keys,
    effect_configs: Optional[dict] = None,
) -> PostProcessingChain:
    """A chain of the effects :data:`EFFECT_REGISTRY` has for ``nuisance_keys``.

    A key naming no effect is skipped, so a caller can pass every nuisance key it has.
    ``effect_configs`` is ``{nuisance key: constructor keyword arguments}``.
    """
    configs = effect_configs or {}
    effects = [
        EFFECT_REGISTRY[key](**configs.get(key, {}))
        for key in nuisance_keys
        if key in EFFECT_REGISTRY
    ]
    return PostProcessingChain(effects)
