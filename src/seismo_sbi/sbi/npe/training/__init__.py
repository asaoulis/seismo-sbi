"""Training and reloading the NPE.

``train`` holds :class:`~seismo_sbi.sbi.npe.training.train.CompressionTrainer` and
``lightning_module`` the Lightning module it trains; ``mmd`` the misspecification-robust MMD
loss; ``legacy_checkpoints`` and ``checkpoint_loading`` load checkpoints written by earlier
versions and by this one.
"""
