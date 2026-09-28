"""Inference pipeline, noise models, compression, inversion and training sets.

``pipeline`` and ``pipeline_variants`` run the method end to end from an ``SBI_Configuration``
(``configuration``); ``noises`` holds the covariances and noise samplers; ``compression`` the
score and learned compressors; ``inversion`` the likelihood sampler, the neural posterior and the
least-squares solver; ``datasets`` and ``training_configuration`` build and feed NPE training
sets; ``types`` holds the parameter and result records.
"""
