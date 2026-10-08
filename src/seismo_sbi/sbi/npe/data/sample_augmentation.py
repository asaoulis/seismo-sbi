"""The training-time augmentation and noise applied to each clean NPE training sample.

:class:`SampleAugmentation` folds the nuisance chain into the clean data, adds one noise draw, and
then applies the post-noise chain, such as component dropout, that must zero a channel exactly.
"""

import numpy as np
import torch

from seismo_sbi.nuisance_effects.post_processing import apply_chain_to_array


class SampleAugmentation:
    """Augmentation and noise for data ``(n_stations, n_components, n_samples)`` laid out by
    ``data_loader``'s receivers and components.

    ``synthetic_noise_model_sampler`` draws the noise; ``augmentation_chain`` acts on the clean
    data before it, ``post_noise_augmentation_chain`` on the noisy data after it, each with its
    ``{nuisance key: activation}`` parameters.
    """

    def __init__(self, data_loader, synthetic_noise_model_sampler, augmentation_chain=None,
                 augmentation_nuisance_params=None, post_noise_augmentation_chain=None,
                 post_noise_nuisance_params=None, torch_dtype=torch.float32):
        self.data_loader = data_loader
        self.synthetic_noise_model_sampler = synthetic_noise_model_sampler
        self.augmentation_chain = augmentation_chain
        self.augmentation_nuisance_params = augmentation_nuisance_params or {}
        self.post_noise_augmentation_chain = post_noise_augmentation_chain
        self.post_noise_nuisance_params = post_noise_nuisance_params or {}
        self.torch_dtype = torch_dtype

    def __call__(self, D):
        """``(x, noise_present)``: the augmented noisy data as a tensor, and the stations the noise
        window carried (``None`` unless the sampler allows incomplete windows)."""
        D = self._augment_clean(D)
        D = torch.as_tensor(D, dtype=self.torch_dtype)
        x, noise_present = self._add_noise(D)
        x = self._augment_noisy(x)
        return x, noise_present

    def _augment_clean(self, D):
        """Apply the nuisance augmentation to the clean data ``(N, C, T)``, before noise is added."""
        if self.augmentation_chain is not None and self.augmentation_chain.effects:
            D = apply_chain_to_array(
                self.augmentation_chain,
                D,
                self.data_loader.receivers,
                self.data_loader.components,
                self.augmentation_nuisance_params,
            )
        return D

    def _add_noise(self, D):
        """``(x, noise_present)``: ``D`` plus one noise draw, and the stations the noise window
        carried (``None`` unless the sampler allows incomplete windows)."""
        draw = self.synthetic_noise_model_sampler.draw()
        noise_present = None if draw.present is None else np.asarray(draw.present, dtype=bool)
        noise = np.asarray(draw.noise)
        noise_rows = self.data_loader.zero_fill_unused_components(
            noise.reshape(-1, D.shape[-1]), D.shape[-1]
        )
        noise_arr = np.asarray(noise_rows)
        x = D + torch.as_tensor(noise_arr, dtype=self.torch_dtype).reshape(*D.shape)
        return x, noise_present

    def _augment_noisy(self, x):
        """Apply the post-noise chain (component dropout) on the full station set."""
        post_chain = self.post_noise_augmentation_chain
        if post_chain is not None and post_chain.effects:
            x_aug = apply_chain_to_array(
                post_chain,
                x.numpy(),
                self.data_loader.receivers,
                self.data_loader.components,
                self.post_noise_nuisance_params,
            )
            x = torch.as_tensor(x_aug, dtype=self.torch_dtype).reshape(*x.shape)
        return x
