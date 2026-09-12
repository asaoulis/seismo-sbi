"""Forward model backed by an ensemble of precomputed Green's-function databases.

One member is drawn per simulation, so repeated simulations of the same source sample the
theory error carried by the spread of the Earth models; ``use_fiducial=True`` selects the
reference member instead. A backend supplies ``members``, ``fiducial_member`` and, if it uses
the member-dispatch pattern, ``_simulate_with_member``.
"""

from abc import ABC, abstractmethod
import numpy as np

from .base import Simulator


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
