from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from .simulator import Simulator
from .wrapper import GenericPointSource, InstaseisDBQuerier


class GFEnsembleSimulator(Simulator, ABC):
    """Simulator backed by an ensemble of precomputed 1D Earth-model GFs.

    One member is drawn per simulation; the fiducial member is used when
    use_fiducial=True.  Subclasses must expose `members` and `fiducial_member`
    and implement `_simulate_with_member` if they rely on the member-dispatch
    pattern (e.g. Instaseis).  CPS subclasses keep their own
    generic_point_source_simulation and call select_member() directly inside
    their GF-loading routine.
    """

    @property
    @abstractmethod
    def members(self) -> list:
        """Ordered list of ensemble members (e.g. folder paths or DB paths)."""
        ...

    @property
    @abstractmethod
    def fiducial_member(self):
        """The fiducial (reference) member of the ensemble."""
        ...

    @property
    def num_models(self) -> int:
        return len(self.members)

    def select_member(self, *, use_fiducial=False, seed=None):
        """Draw one member from the ensemble.

        Reproduces the legacy CPS call sequence exactly:
        np.random.seed(seed) then np.random.choice(members).
        """
        if use_fiducial:
            return self.fiducial_member
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(self.members)

    def _simulate_with_member(self, member, source, **kwargs) -> dict:
        """Simulate seismograms using a specific ensemble member.

        Override in subclasses that use the select_member →
        _simulate_with_member dispatch pattern (e.g. InstaseisEnsembleSimulator).
        CPS subclasses integrate member selection into their GF loading instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _simulate_with_member "
            "to use the member-dispatch pattern."
        )


class InstaseisEnsembleSimulator(GFEnsembleSimulator):
    """Instaseis simulator backed by an ensemble of Instaseis databases.

    One DB is drawn at random per simulation (using select_member), matching
    the CPS ensemble methodology.

    Parameters
    ----------
    instaseis_ensemble_dir : str or Path
        Directory whose immediate subdirectories are each a full Instaseis DB.
    instaseis_fiducial_loc : str or Path
        Path to the fiducial (reference) Instaseis DB.
    """

    def __init__(self, instaseis_ensemble_dir, instaseis_fiducial_loc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ensemble_dir = Path(instaseis_ensemble_dir)
        self._members = sorted(
            [str(p) for p in ensemble_dir.iterdir() if p.is_dir()]
        )
        if not self._members:
            raise FileNotFoundError(
                f"No Instaseis DB directories found in {ensemble_dir}"
            )
        self._fiducial_member = str(instaseis_fiducial_loc)
        self._db_cache: dict[str, InstaseisDBQuerier] = {}

        # Derive sampling_rate from the fiducial DB (matches InstaseisSourceSimulator)
        self.sampling_rate = float(
            self._querier_for(self._fiducial_member).sampling_rate
        )

    @property
    def members(self) -> list:
        return self._members

    @property
    def fiducial_member(self):
        return self._fiducial_member

    def _querier_for(self, db_path) -> InstaseisDBQuerier:
        # Convert numpy.str_ (returned by np.random.choice on string lists) to
        # plain Python str to avoid "Can't mix strings and bytes" in os.walk
        # inside instaseis.open_db.
        key = str(db_path)
        if key not in self._db_cache:
            self._db_cache[key] = InstaseisDBQuerier(
                key, self.synthetics_processing, self.seismogram_length
            )
        return self._db_cache[key]

    def _simulate_with_member(self, member: str, source: GenericPointSource, *,
                              stf_duration=None, **kwargs) -> dict:
        querier = self._querier_for(member)
        all_seismograms_map = {}
        for receiver in self.receivers.iterate():
            all_seismograms_map[receiver.station_name] = {}
            receiver_results = querier.get_seismograms(
                source, receiver, self.components, stf_duration=stf_duration
            )
            for component in self.components:
                all_seismograms_map[receiver.station_name][component] = (
                    receiver_results[component]
                )
        return all_seismograms_map

    def generic_point_source_simulation(
        self, source: GenericPointSource, *, use_fiducial=False, seed=None,
        stf_duration=None, **kwargs
    ) -> dict:
        member = self.select_member(use_fiducial=use_fiducial, seed=seed)
        return self._simulate_with_member(member, source, stf_duration=stf_duration)
