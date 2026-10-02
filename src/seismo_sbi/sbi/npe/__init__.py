"""Neural posterior estimation on raw waveforms.

``networks`` holds the embedding net that turns station traces into a summary, ``maf`` the
conditional flow over the source parameters, ``data`` the datasets and loaders that feed
simulations with training-time augmentation, and ``training`` the trainer and its checkpoints.
``source_conditioning`` packs the observation with its station and source geometry, and
``posterior_sampling`` draws posteriors for chosen station subsets.
"""
