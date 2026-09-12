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
    """Simulator backed by an ensemble of precomputed one-dimensional Earth models.

    A subclass exposes ``members`` and ``fiducial_member``. It then either implements
    ``_simulate_with_member``, or calls :meth:`select_member` inside its own
    ``generic_point_source_simulation``, as the CPS backends do.
    """

    @property
    @abstractmethod
    def members(self) -> list:
        """Ensemble members in a fixed order, usually database paths."""
        ...

    @property
    @abstractmethod
    def fiducial_member(self):
        """The reference member of the ensemble."""
        ...

    @property
    def num_models(self) -> int:
        return len(self.members)

    def select_member(self, *, use_fiducial=False, seed=None):
        """One member drawn from the ensemble.

        A seed is applied as ``np.random.seed(seed)`` then ``np.random.choice(members)``.
        """
        if use_fiducial:
            return self.fiducial_member
        if seed is not None:
            np.random.seed(seed)
        return np.random.choice(self.members)

    def _simulate_with_member(self, member, source, **kwargs) -> dict:
        """``{station: {component: waveform}}`` simulated on one ensemble member."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _simulate_with_member "
            "to use the member-dispatch pattern."
        )
