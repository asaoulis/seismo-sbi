"""Noise samplers: the noise a training sample or a synthetic test event gets.

Every sampler is a :class:`NoiseSampler` returning a :class:`NoiseDraw`. ``WhiteNoiseSampler``
draws independent Gaussian noise of one level, ``GaussianNoiseSampler`` draws from Cholesky
factors of block-diagonal covariances and can rescale each block to a measured variance, and
``BlockGaussianSampler`` draws from precomputed factors.
"""
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import numpy as np
from scipy.linalg import toeplitz

from seismo_sbi.sbi.noises.covariance_base import pre_event_variances, station_component_value


class NoiseDraw(NamedTuple):
    """One noise realisation.

    ``noise`` is ``(data_vector_length,)``. ``present`` ``(n_stations,)`` marks the stations a
    recorded window holds, or is None unless the sampler draws incomplete windows.
    ``covariance_data`` describes the noise, or is None when the sampler holds none: a recorded
    window's ``{station: {component: autocovariance}}`` (lag 0 the variance), or the data a
    Gaussian covariance was built from.
    """
    noise: np.ndarray
    present: Optional[np.ndarray] = None
    covariance_data: Optional[object] = None


class NoiseSampler(ABC):
    """A source of data-vector noise: :meth:`draw` for a training sample, :meth:`draw_with_covariance`
    for a synthetic test event, whose inversion needs the covariance describing its noise."""

    @abstractmethod
    def draw(self) -> NoiseDraw:
        """One noise realisation for a training sample."""

    def draw_with_covariance(self) -> NoiseDraw:
        """One noise realisation with the covariance data that describes it."""
        return self.draw()


class WhiteNoiseSampler(NoiseSampler):
    """Independent Gaussian noise of standard deviation ``noise_level`` on each of
    ``data_vector_length`` samples."""

    def __init__(self, noise_level, data_vector_length):
        self.noise_level = noise_level
        self.data_vector_length = data_vector_length

    def draw(self):
        return NoiseDraw(np.random.normal(0, self.noise_level * np.ones((self.data_vector_length))))


class GaussianNoiseSampler(NoiseSampler):
    """Gaussian sampler that owns covariance and can be adapted.

    This class is intended mainly for the block-diagonal covariances where each
    block is Toeplitz. It keeps track of

    - receivers: layout and component ordering
    - toeplitz_cols: first column per block (shape [n_blocks, block_size])
    - cov_blocks: full covariance blocks (shape [n_blocks, block_size, block_size])
    - Ls: Cholesky factors for each block, recomputed when covariance changes

    :meth:`rescale_to` rescales each Toeplitz column so that col[0] matches a
    measured pre-event variance, and updates cov_blocks and Ls accordingly.
    """

    def __init__(self, receivers, data_vector_length,
                 toeplitz_cols=None,
                 cov_blocks=None,
                 station_component_covariances=None):
        self.receivers = receivers
        self.data_vector_length = data_vector_length
        self.station_component_covariances = station_component_covariances

        self.receiver_components = [] if receivers is None else [
            (receiver.station_name, component)
            for receiver in self.receivers.iterate()
            for component in receiver.components
        ]

        if toeplitz_cols is not None:
            self.toeplitz_cols = np.asarray(toeplitz_cols, dtype=float)
            self.cov_blocks = np.asarray(
                cov_blocks
                if cov_blocks is not None
                else [toeplitz(c) for c in self.toeplitz_cols],
                dtype=float,
            )
        elif cov_blocks is not None:
            self.cov_blocks = np.asarray(cov_blocks, dtype=float)
            # Treat diagonal as Toeplitz first column if nothing else given.
            self.toeplitz_cols = np.array(
                [block[0, :].copy() for block in self.cov_blocks],
                dtype=float,
            )
        else:
            raise ValueError("Either toeplitz_cols or cov_blocks must be provided")

        self._build_cholesky()

    def _build_cholesky(self):
        """Build Cholesky factors for all covariance blocks."""
        Ls = []
        for block in self.cov_blocks:
            # Assume blocks are already regularised / psd further up.
            Ls.append(np.linalg.cholesky(block))
        self.Ls = Ls
        self.block_sizes = [L.shape[0] for L in self.Ls]

    def draw(self):
        """A draw from the current covariance, with ``station_component_covariances`` (None if the
        sampler was not given it) as its covariance data."""
        return NoiseDraw(draw_block_noise(self.Ls, self.block_sizes), None, self.station_component_covariances)

    def rescale_to(self, covariance_data):
        """Rescale each block to the pre-event variance in ``covariance_data``
        ``{station: {component: autocovariance}}``, keeping its correlation structure.

        The Toeplitz column ``c`` of each (station, component) block is scaled by ``s`` with
        ``(s * c)[0]`` equal to the target variance, which scales the whole block.
        """
        if self.toeplitz_cols is None or self.cov_blocks is None or not self.receiver_components:
            return

        target_variances = pre_event_variances(covariance_data)
        scaled_cols = []
        for (station, comp), col in zip(self.receiver_components, self.toeplitz_cols):
            c0 = col[0]
            if c0 == 0:
                scaled_cols.append(col)
                continue
            target_var = float(station_component_value(target_variances, station, comp))
            scale = target_var / c0
            scaled_cols.append(col * scale)
        self.toeplitz_cols = np.asarray(scaled_cols, dtype=float)

        self.cov_blocks = np.asarray([toeplitz(c) for c in self.toeplitz_cols], dtype=float)
        self._build_cholesky()


class BlockGaussianSampler(NoiseSampler):
    """Block-diagonal Gaussian noise from precomputed Cholesky factors ``Ls`` of blocks of ``block_sizes``."""

    def __init__(self, Ls, block_sizes, station_component_covariances):
        self.Ls = Ls
        self.block_sizes = block_sizes
        self.station_component_covariances = station_component_covariances

    def draw(self):
        return NoiseDraw(draw_block_noise(self.Ls, self.block_sizes), None, self.station_component_covariances)


def draw_block_noise(cholesky_factors, block_sizes):
    """One draw of block-diagonal Gaussian noise: each block's Cholesky factor times a standard normal."""
    noise_vectors = []
    for L, n in zip(cholesky_factors, block_sizes):
        z = np.random.randn(n)
        noise_vectors.append(L @ z)
    return np.concatenate(noise_vectors)
