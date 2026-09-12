"""Simulators that serve disjoint receiver subsets from different velocity models.

A :class:`MultiModelSimulator` dispatches each receiver subset to its own per-region
sub-simulator and merges the outputs, so from outside it behaves like one simulator over the
union of the receivers. The use case is path-specific theory error: receivers on one side of the
array are served by one velocity-model ensemble and the rest by another, so a single forward
simulation bakes in a regionally varying theory error.
"""

from abc import ABC, abstractmethod

from .base import Simulator
from .sources import GenericPointSource


class MultiModelSimulator(Simulator, ABC):
    """Dispatch receiver subsets to per-region sub-simulators and merge outputs.

    Holds a list of ``(sub_receivers, sub_simulator)`` where each sub-simulator
    is responsible for a *subset* of the global receiver geometry.  All
    sub-simulators share the global ``components`` / processing configuration.

    Parameters
    ----------
    models : list[dict]
        One dict per region.  Each must contain ``"receivers"`` (a
        :class:`Receivers` subset) and either a pre-built ``"simulator"`` or the
        backend-specific keys consumed by :meth:`_build_sub_simulator`.

    Notes
    -----
    * Post-processing (amplitude error, dropout, time shifts) is applied ONCE at
      the union level by the inherited :meth:`Simulator.run_simulation`, so the
      sub-simulators must be constructed WITHOUT post-processing effects.
    * **Member draws are independent per region.**  With no seed, each
      sub-simulator draws from the shared RNG independently; with an explicit
      seed, sub-model ``i`` is given ``seed + i`` so draws stay reproducible yet
      decorrelated across regions.  (In production the forward path passes no
      seed, so each region draws an independent member every simulation.)
    """

    def __init__(self, models, *args, **kwargs):
        # Cooperative init: forwards components/receivers/duration/processing/
        # post_processing_effects (and, for the CPS subclass, cps_path) down the
        # MRO to the backend base and finally Simulator.__init__.
        super().__init__(*args, **kwargs)
        self.sub_sims, self.sub_receivers = self._init_sub_models(models)
        # Per-region member count (regions are equal-sized ensembles); retained
        # for parity with the historical single-class MultiModelCPSSimulator.
        self.num_models = self.sub_sims[0].num_models if self.sub_sims else 0

    def _init_sub_models(self, models):
        if not isinstance(models, (list, tuple)) or len(models) == 0:
            raise ValueError(
                "'models' must be a non-empty list/tuple of configuration dicts."
            )
        sub_sims = []
        sub_receivers = []
        for cfg in models:
            if not isinstance(cfg, dict):
                raise TypeError(
                    "Each element of 'models' must be a dict with at least "
                    "'receivers' and the backend sub-model configuration."
                )
            receivers = cfg.get("receivers")
            if receivers is None:
                raise KeyError(
                    f"{type(self).__name__} config dict missing required key 'receivers'."
                )
            existing = cfg.get("simulator")
            sim = existing if existing is not None else self._build_sub_simulator(cfg, receivers)
            sub_sims.append(sim)
            sub_receivers.append(receivers)
        return sub_sims, sub_receivers

    @abstractmethod
    def _build_sub_simulator(self, cfg: dict, sub_receivers) -> Simulator:
        """Construct a backend-specific sub-simulator over ``sub_receivers``."""
        ...

    @staticmethod
    def _sub_model_seed(seed, model_index):
        """Independent-per-region draw: offset the seed so each region draws a
        decorrelated (but reproducible) member.  ``seed=None`` stays ``None``
        (unseeded => independent draws from the shared RNG)."""
        if seed is None:
            return None
        return seed + model_index

    def generic_point_source_simulation(self, source: GenericPointSource, *,
                                        seed=None, **kwargs) -> dict:
        """Run each sub-simulator on its receiver subset and merge per-station.

        The returned dict has the same structure as a single :class:`Simulator`
        over ``self.receivers``.
        """
        per_model_results = []
        for model_index, (sim, sub_rec) in enumerate(zip(self.sub_sims, self.sub_receivers)):
            sub_kwargs = dict(kwargs)
            sub_seed = self._sub_model_seed(seed, model_index)
            # Only inject `seed` when explicitly requested so the production
            # (unseeded) path is byte-identical to the historical single-class
            # behaviour and never relies on a sub-simulator accepting `seed`.
            if sub_seed is not None:
                sub_kwargs["seed"] = sub_seed
            sub_map = sim.generic_point_source_simulation(source, **sub_kwargs)
            per_model_results.append((sub_rec, sub_map))

        all_seismograms_map = {}
        for rec in self.receivers.iterate():
            station_name = rec.station_name
            all_seismograms_map[station_name] = {}

            owning_map = None
            for sub_rec, sub_map in per_model_results:
                if any(r.station_name == station_name for r in sub_rec.iterate()):
                    owning_map = sub_map
                    break
            if owning_map is None:
                raise KeyError(
                    f"Station '{station_name}' in global receivers is not covered "
                    "by any sub-model configuration."
                )

            for comp in rec.components:
                try:
                    all_seismograms_map[station_name][comp] = owning_map[station_name][comp]
                except KeyError as exc:
                    raise KeyError(
                        f"Component '{comp}' for station '{station_name}' missing "
                        "from sub-model output. Ensure components are consistent."
                    ) from exc

        return all_seismograms_map
