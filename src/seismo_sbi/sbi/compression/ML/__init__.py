"""Neural compression and NPE training.

``seismogram_transformer`` and its encoders (``station_encoders``, ``axial_transformer``,
``pma_pooling``) embed the station traces; ``maf`` builds the flow; ``dataloading`` feeds
simulations with augmentation; ``train`` holds :class:`~.train.CompressionTrainer`.
"""
