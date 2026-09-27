"""Zeroed stations and zeroed channels: data that is missing at random.

``InstrumentDropoutEffect`` zeroes whole stations; ``ComponentDropoutEffect`` zeroes single
channels after noise is added, always keeping one per station.
"""
from __future__ import annotations

import numpy as np

from seismo_sbi.nuisance_effects.seismogram_effect import SeismogramEffect


class InstrumentDropoutEffect(SeismogramEffect):
    """Zero a whole station's traces, independently per station.

    Nuisance key ``instrument_dropout``: the probability in ``[0, 1]`` that a station is
    zeroed. Absent or zero is the identity.
    """

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        instrument_dropout: float | None = None,
        **_ignored,
    ) -> dict:
        if instrument_dropout is None:
            return seismograms_map

        def _zero(components):
            return {comp: np.zeros_like(trace, dtype=np.float64) for comp, trace in components.items()}

        return self._apply_per_station_gated(seismograms_map, instrument_dropout, _zero)


class ComponentDropoutEffect(SeismogramEffect):
    """Zero individual present channels of a station, modelling an event missing a subset
    of them.

    Nuisance key ``component_dropout``: the probability each present channel is dropped,
    drawn independently per channel in ``receivers.iterate()`` order. At least one channel
    per station is always kept, a whole absent station being
    :class:`InstrumentDropoutEffect`'s business.

    It must run after sensor noise is added, so a dropped channel is exactly zero as a
    genuinely absent one is, which is why it is staged post-noise and never baked into a
    simulation.
    """

    def __call__(
        self,
        seismograms_map: dict,
        receivers,
        *,
        component_dropout: float | None = None,
        **_ignored,
    ) -> dict:
        if component_dropout is None:
            return seismograms_map

        p = float(np.clip(component_dropout, 0.0, 1.0))
        # From the receivers, not the map keys, which also carry zero-filled absent
        # components the adapter inserts.
        present_by_station = {rec.station_name: list(rec.components) for rec in receivers.iterate()}

        result = {}
        for station, components in seismograms_map.items():
            new_components = {comp: trace.astype(np.float64) for comp, trace in components.items()}
            present = [c for c in present_by_station.get(station, []) if c in new_components]
            if len(present) >= 2:
                drop = [c for c in present if np.random.uniform() < p]
                # One channel is restored at random if every one was selected, so the
                # station never becomes all-zero.
                if len(drop) == len(present):
                    keep = present[np.random.randint(len(present))]
                    drop = [c for c in drop if c != keep]
                for c in drop:
                    new_components[c] = np.zeros_like(new_components[c], dtype=np.float64)
            result[station] = new_components
        return result
