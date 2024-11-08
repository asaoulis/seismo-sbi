from typing import NamedTuple, Callable, Dict, Tuple
from seismo_sbi.sbi.compression.gaussian import ScoreCompressionData
import numpy as np

class JobData(NamedTuple):

    job_name: str
    noise_type: str
    data_vector: np.ndarray
    theta0: Dict
    covariance: Dict = None
    priors: Tuple = (None, None)

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
