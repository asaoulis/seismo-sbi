"""Training sets for the inference methods.

``dataset_generator`` draws source parameters from the prior and simulates them in parallel;
``dataset_compressor`` adds noise to each simulation and compresses it; ``training_data``
assembles the pipeline, simulations, noise model, scaler and augmentation an NPE run consumes.
"""
