"""The post-processing chain: effects applied in order to synthetic seismograms.

``PostProcessingChain`` composes ``SeismogramEffect`` objects, each one's output feeding the
next. ``build_post_processing_chain(nuisance_keys)`` assembles a chain from ``EFFECT_REGISTRY``,
silently skipping keys that name no effect, so a caller can pass every nuisance key it has.
The augmentation builders and ``apply_chain_to_array`` run the same effects in the dataloader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from seismo_sbi.nuisance_effects.amplitude_effect import AmplitudeErrorEffect
from seismo_sbi.nuisance_effects.anisotropy_effects import AzimuthalAnisotropyEffect, ShearSplittingEffect
from seismo_sbi.nuisance_effects.dispersion_effect import DispersionSpreadEffect
from seismo_sbi.nuisance_effects.dropout_effects import ComponentDropoutEffect, InstrumentDropoutEffect
from seismo_sbi.nuisance_effects.scattering_coda_effect import ScatteringCodaEffect
from seismo_sbi.nuisance_effects.seismogram_effect import SeismogramEffect
from seismo_sbi.nuisance_effects.time_shift_effect import TimeShiftErrorEffect


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


@dataclass(frozen=True)
class NuisanceEffect:
    """How one nuisance key acts: the effect class built for it, the stages it may run at
    (``"simulation"``, ``"training_augmentation"``, ``"training_augmentation_post_noise"``), and
    whether its constructor takes the ``sampling_rate`` in samples per second.
    """

    effect: type
    stages: tuple
    needs_sampling_rate: bool = False


_SIMULATED_OR_AUGMENTED = ("simulation", "training_augmentation")

#: Nuisance parameter key to the effect it builds; the one table every stage reads.
EFFECT_REGISTRY: dict[str, NuisanceEffect] = {
    "amplitude_error": NuisanceEffect(AmplitudeErrorEffect, _SIMULATED_OR_AUGMENTED),
    "instrument_dropout": NuisanceEffect(InstrumentDropoutEffect, _SIMULATED_OR_AUGMENTED),
    "time_shift_error": NuisanceEffect(TimeShiftErrorEffect, _SIMULATED_OR_AUGMENTED, True),
    "scattering_coda": NuisanceEffect(ScatteringCodaEffect, _SIMULATED_OR_AUGMENTED),
    # Must run after the noise is added, so a dropped channel is exactly zero.
    "component_dropout": NuisanceEffect(ComponentDropoutEffect, ("training_augmentation_post_noise",)),
    "azimuthal_anisotropy": NuisanceEffect(AzimuthalAnisotropyEffect, ("simulation",), True),
    "shear_wave_splitting": NuisanceEffect(ShearSplittingEffect, ("simulation",), True),
    "dispersion_spread": NuisanceEffect(DispersionSpreadEffect, ("simulation",), True),
}


#: Nuisance keys that augment the source-location conditioning vector rather than the
#: waveform, so they have no effect class and the dataloader applies them.
CONDITIONING_AUGMENTABLE_KEYS: tuple[str, ...] = (
    "source_location_error",
)


def effect_keys_at(stage: str) -> tuple:
    """The nuisance keys whose effect may run at ``stage``."""
    return tuple(key for key, entry in EFFECT_REGISTRY.items() if stage in entry.stages)


def with_sampling_rate(nuisance_keys, effect_configs: Optional[dict], sampling_rate) -> dict:
    """``effect_configs`` with ``sampling_rate`` added for every key in ``nuisance_keys`` whose
    effect takes it; the caller's dicts are not changed. A ``sampling_rate`` of None adds nothing.
    """
    configs = dict(effect_configs or {})
    if sampling_rate is None:
        return configs
    for key in nuisance_keys:
        if key in EFFECT_REGISTRY and EFFECT_REGISTRY[key].needs_sampling_rate:
            configs[key] = {**configs.get(key, {}), "sampling_rate": sampling_rate}
    return configs


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
    in samples per second is passed to the effects that take it. An empty selection gives an empty
    chain, which the dataloader treats as no augmentation.
    """
    eligible = effect_keys_at(stage)
    aug_keys = [
        key for key in nuisance
        if key in eligible
        and nuisance_stage.get(key, "simulation") == stage
    ]
    configs = with_sampling_rate(aug_keys, effect_configs, sampling_rate)

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
        EFFECT_REGISTRY[key].effect(**configs.get(key, {}))
        for key in nuisance_keys
        if key in EFFECT_REGISTRY
    ]
    return PostProcessingChain(effects)
