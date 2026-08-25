"""Multi-model (multi-region) simulators.

A :class:`MultiModelSimulator` dispatches *disjoint* receiver subsets to
different per-region sub-simulators, then merges their outputs so that from the
outside it behaves like a single :class:`Simulator` over the *union* of all
receivers.  This is the generic, backend-agnostic base that mirrors the
ensemble layering (:class:`GFEnsembleSimulator` -> Instaseis/CPS subclasses):
the per-station merge lives here once, and subclasses only specialise how each
backend-specific sub-simulator is constructed.

Two specialisations live in the codebase:
  - :class:`InstaseisMultiModelSimulator` (here) — each region backed by its own
    Instaseis-DB ensemble (:class:`InstaseisEnsembleSimulator`).
  - ``MultiModelCPSSimulator`` (in ``cps_simulator/simulator.py``) — each region
    backed by a CPS precomputed Green's-function model.

The canonical use case is path-specific theory error: on-island Santorini
receivers are served by a "mode A" velocity-model ensemble while every other
receiver is served by a "mode B" ensemble, so a single forward simulation bakes
the regionally-varying theory error.
"""

from abc import ABC, abstractmethod

from .simulator import Simulator
from .ensemble import InstaseisEnsembleSimulator
from .wrapper import GenericPointSource


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


class InstaseisMultiModelSimulator(MultiModelSimulator):
    """Multi-region simulator backed by per-region Instaseis-DB ensembles.

    Each sub-model config dict provides either a pre-built ``"simulator"`` (an
    :class:`InstaseisEnsembleSimulator`) or the paths to build one:

    - ``"receivers"``  : a :class:`Receivers` subset for this region,
    - ``"ensemble_dir"``: directory of member Instaseis DBs (drawn per sim),
    - ``"fiducial_dir"``: the fiducial (reference) DB (used when
      ``use_fiducial=True``).

    With ``use_fiducial=True`` each region routes its receivers to ITS OWN
    fiducial DB (per-mode fiducial), so the merged output is the regionally
    consistent reference seismogram.

    ``resample_member_per_station`` is forwarded to every region's
    :class:`InstaseisEnsembleSimulator` (each region then independently draws a
    fresh member per station — the intra-ensemble / per-station theory-error
    mode). Pre-built ``"simulator"`` entries keep whatever flag they were built
    with.
    """

    def __init__(self, models, *args, resample_member_per_station=False, member_sampling=None,
                 sector_lambda=None, **kwargs):
        # Set BEFORE super().__init__: MultiModelSimulator.__init__ builds the sub-simulators
        # (via _init_sub_models -> _build_sub_simulator) inside its own __init__, and
        # _build_sub_simulator reads this flag to forward it into each region's ensemble.
        self.resample_member_per_station = resample_member_per_station
        self.member_sampling = member_sampling
        self.sector_lambda = sector_lambda
        super().__init__(models, *args, **kwargs)
        # Parity with InstaseisEnsembleSimulator / InstaseisSourceSimulator:
        # expose a sampling_rate (all regions share period/sampling).  Optional
        # via getattr so dependency-free mock sub-sims (no DB) still construct.
        self.sampling_rate = getattr(self.sub_sims[0], "sampling_rate", None)

    def _build_sub_simulator(self, cfg, sub_receivers):
        try:
            ensemble_dir = cfg["ensemble_dir"]
            fiducial_dir = cfg["fiducial_dir"]
        except KeyError as exc:
            raise KeyError(
                "Each InstaseisMultiModelSimulator config dict must contain "
                "either 'simulator' or both 'ensemble_dir' and 'fiducial_dir'."
            ) from exc
        return InstaseisEnsembleSimulator(
            instaseis_ensemble_dir=ensemble_dir,
            instaseis_fiducial_loc=fiducial_dir,
            components=self.components,
            receivers=sub_receivers,
            seismogram_duration_in_s=self.seismogram_length,
            synthetics_processing=self.synthetics_processing,
            # Parent applies the post-processing chain once over the union.
            post_processing_effects=[],
            resample_member_per_station=self.resample_member_per_station,
            member_sampling=getattr(self, 'member_sampling', None),
            sector_lambda=getattr(self, 'sector_lambda', None),
        )
