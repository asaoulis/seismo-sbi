"""Multi-model Instaseis simulator: one Instaseis database ensemble per receiver region.

Each sub-model dict carries ``receivers``, ``ensemble_dir`` and ``fiducial_dir``; the merge over
regions lives in :class:`~seismo_sbi.simulators.multi_region.MultiModelSimulator`.
"""

from ..multi_region import MultiModelSimulator
from .ensemble import InstaseisEnsembleSimulator


class InstaseisMultiModelSimulator(MultiModelSimulator):
    """Multi-region simulator backed by one Instaseis database ensemble per region.

    Each sub-model dict carries ``"receivers"`` and either a pre-built ``"simulator"`` or
    ``"ensemble_dir"`` and ``"fiducial_dir"``. Under ``use_fiducial=True`` each region uses its
    own reference database, so the merged output is regionally consistent.
    ``resample_member_per_station`` is forwarded to every region built here; a pre-built
    simulator keeps the flag it was built with.
    """

    def __init__(self, models, *args, resample_member_per_station=False, member_sampling=None,
                 sector_lambda=None, **kwargs):
        # Set before super().__init__, which builds the sub-simulators and reads these.
        self.resample_member_per_station = resample_member_per_station
        self.member_sampling = member_sampling
        self.sector_lambda = sector_lambda
        super().__init__(models, *args, **kwargs)
        # Every region shares the sampling rate; via getattr so a mock sub-simulator with no
        # database still constructs.
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
            post_processing_effects=[],
            resample_member_per_station=self.resample_member_per_station,
            member_sampling=getattr(self, 'member_sampling', None),
            sector_lambda=getattr(self, 'sector_lambda', None),
            source_depth_offset_km=getattr(self, 'source_depth_offset_km', 0.0),
        )
