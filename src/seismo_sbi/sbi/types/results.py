"""Records for jobs and inversion results.

:class:`JobData` holds one job's data vector and truth; :class:`InversionConfig`,
:class:`InversionData` and :class:`InversionResult` hold what an inversion used and returned.
"""

from typing import NamedTuple, Callable, Dict, Optional
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
from seismo_sbi.sbi.inversion.gaussian_prior import GaussianPrior
import numpy as np

class JobData(NamedTuple):
    """One job: its data vector under one test noise, the true parameters (None for a real event),
    the covariance data of its noise, and an optional Gaussian prior on its source parameters."""

    job_name: str
    noise_type: str
    data_vector: np.ndarray
    theta0: Dict
    covariance: Dict = None
    prior: Optional[GaussianPrior] = None

class InversionData(NamedTuple):

    theta0 : np.ndarray
    samples: np.ndarray
    data_scaler: Callable
    compression_data: ScoreCompressionData = None

class InversionConfig(NamedTuple):
    train_noise: str
    test_noise: str
    inversion_method: str

class InversionResult(NamedTuple):

    event_name: str
    inversion_data: InversionData
    inversion_config: InversionConfig

    def __hash__(self):
        return hash((self.event_name, self.inversion_config.train_noise, self.inversion_config.test_noise, self.inversion_config.inversion_method))

class JobResult(NamedTuple):
    compressed_dataset: np.ndarray
    compressed_job: np.ndarray
    data_scaler: Callable
