"""Multi-model Instaseis simulator: one Instaseis database ensemble per receiver region.

Each sub-model dict carries ``receivers``, ``ensemble_dir`` and ``fiducial_dir``; the merge over
regions lives in :class:`~seismo_sbi.simulators.multi_model.MultiModelSimulator`.
"""

from ..multi_model import MultiModelSimulator
from .ensemble import InstaseisEnsembleSimulator


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
            source_depth_offset_km=getattr(self, 'source_depth_offset_km', 0.0),
        )
