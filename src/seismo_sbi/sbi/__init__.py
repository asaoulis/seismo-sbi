"""Inference pipeline, noise models, compression and training.

``pipeline`` and ``pipeline_variants`` run the method end to end from an ``SBI_Configuration``
(``configuration``); ``noises`` holds the covariances and noise samplers; ``compression`` the
score and learned compressors; ``training_configuration`` and ``training_data`` feed NPE
training; ``types`` holds the parameter and result records.
"""
