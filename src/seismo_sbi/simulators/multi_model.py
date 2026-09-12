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

    ``models`` is one dict per region, each carrying ``"receivers"`` and either a pre-built
    ``"simulator"`` or the backend keys :meth:`_build_sub_simulator` consumes. Every region
    shares the global components and processing configuration.

    Post-processing is applied once over the union of the receivers by the inherited
    :meth:`Simulator.run_simulation`, so sub-simulators are built without effects. Member
    draws are independent per region: unseeded from the shared generator, seeded with
    ``seed + i`` for region ``i``.
    """

    def __init__(self, models, *args, **kwargs):
        # Cooperative init: the keyword arguments travel down the MRO to Simulator.__init__.
        super().__init__(*args, **kwargs)
        self.sub_sims, self.sub_receivers = self._init_sub_models(models)
        # Per-region member count; the regions are equal-sized ensembles.
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
        """The seed region ``index`` draws with, or ``None`` to draw from the shared generator."""
        if seed is None:
            return None
        return seed + model_index

    def generic_point_source_simulation(self, source: GenericPointSource, *,
                                        seed=None, **kwargs) -> dict:
        """``{station: {component: waveform}}`` over the union of every region's receivers."""
        per_model_results = []
        for model_index, (sim, sub_rec) in enumerate(zip(self.sub_sims, self.sub_receivers)):
            sub_kwargs = dict(kwargs)
            sub_seed = self._sub_model_seed(seed, model_index)
            # Passed only when asked for, so a sub-simulator need not accept a seed at all.
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
