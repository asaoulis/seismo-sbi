"""Gaussian noise samplers drawing from block-diagonal covariances.

``GaussianNoiseSampler`` draws from Cholesky factors of the per-trace blocks and can rescale each
block to a measured variance; ``BlockGaussianSampler`` draws from precomputed factors.
"""
import numpy as np
from scipy.linalg import toeplitz


class GaussianNoiseSampler:
    """Gaussian sampler that owns covariance and can be adapted.

    This class is intended mainly for the block-diagonal covariances where each
    block is Toeplitz. It keeps track of

    - receivers: layout and component ordering
    - toeplitz_cols: first column per block (shape [n_blocks, block_size])
    - cov_blocks: full covariance blocks (shape [n_blocks, block_size, block_size])
    - Ls: Cholesky factors for each block, recomputed when covariance changes

    set_adaptive_covariance_with_misc_data(misc_data) expects misc_data to be a
    nested dict misc_data[station][component] with *actual* variance per
    station-component. It rescales each Toeplitz column so that col[0] matches
    the new variance, and updates cov_blocks and Ls accordingly.
    """

    def __init__(self, receivers, data_vector_length,
                 toeplitz_cols=None,
                 cov_blocks=None,
                 station_component_covariances=None):
        self.receivers = receivers
        self.data_vector_length = data_vector_length
        self.station_component_covariances = station_component_covariances

        # Build (station, component) list in the same order as BlockDiagonal*.
        self.receiver_components = [
            (receiver.station_name, component)
            for receiver in self.receivers.iterate()
            for component in receiver.components
        ]

        # Normalise internal representation.
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

    # Internal helpers
    def _build_cholesky(self):
        """Build Cholesky factors for all covariance blocks."""
        Ls = []
        for block in self.cov_blocks:
            # Assume blocks are already regularised / psd further up.
            Ls.append(np.linalg.cholesky(block))
        self.Ls = Ls
        self.block_sizes = [L.shape[0] for L in self.Ls]

    def _get_target_variance(self, misc_data, station, component):
        """Extract scalar variance from misc_data[station][component]."""
        try:
            val = misc_data[station][component]
        except KeyError:
            # Handle "E"/"N" vs "1"/"2" naming.
            mapped = component.replace("E", "1").replace("N", "2")
            val = misc_data[station][mapped]
        if hasattr(val, "size") and val.size > 1:
            return float(val.ravel()[0])
        return float(val)

    # Public API
    def __call__(self, *args, **kwargs):
        """Draw a sample from the current covariance.

        Returns
        -------
        noise_vector : np.ndarray, shape (n_total,)
        meta : any
            For compatibility with previous interfaces we return
            station_component_covariances as the second value,
            if available, else None.
        """
        noise_vectors = []
        for L, n in zip(self.Ls, self.block_sizes):
            z = np.random.randn(n)
            noise_vectors.append(L @ z)
        noise_vector = np.concatenate(noise_vectors)
        return noise_vector, self.station_component_covariances

    def set_adaptive_covariance_with_misc_data(self, misc_data):
        """Adapt Toeplitz columns and covariance blocks using misc_data.

        For each block (station, component), let c be its Toeplitz first
        column. We compute a scale factor s such that

            (s * c)[0] == target_variance

        where target_variance is read from misc_data[station][component]. The
        entire Toeplitz column is scaled by s, which scales the full block and
        hence preserves its correlation structure while adjusting its marginal
        variance.
        """
        if self.toeplitz_cols is None or self.cov_blocks is None:
            return

        scaled_cols = []
        for (station, comp), col in zip(self.receiver_components, self.toeplitz_cols):
            c0 = col[0]
            if c0 == 0:
                scaled_cols.append(col)
                continue
            target_var = self._get_target_variance(misc_data, station, comp)
            scale = target_var / c0
            scaled_cols.append(col * scale)
        self.toeplitz_cols = np.asarray(scaled_cols, dtype=float)

        # Rebuild covariance blocks from scaled Toeplitz columns.
        self.cov_blocks = np.asarray([toeplitz(c) for c in self.toeplitz_cols], dtype=float)
        # Rebuild Cholesky factors to update the sampler dynamically.
        self._build_cholesky()


class BlockGaussianSampler:
    def __init__(self, Ls, block_sizes, station_component_covariances):
        self.Ls = Ls
        self.block_sizes = block_sizes
        self.station_component_covariances = station_component_covariances

    def __call__(self, *args, **kwargs):
        noise_vectors = []
        for L, n in zip(self.Ls, self.block_sizes):
            z = np.random.randn(n)
            noise_vectors.append(L @ z)
        noise_vector = np.concatenate(noise_vectors)
        return noise_vector, self.station_component_covariances
