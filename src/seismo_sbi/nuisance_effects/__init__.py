"""Nuisance effects: what a real recording does to a synthetic seismogram.

``seismogram_effect.SeismogramEffect`` is the base class; each ``*_effect(s)`` module holds one
family (amplitude, dropout, time shift, scattering coda, anisotropy, dispersion) and
``lanczos_shift`` their sub-sample shifts. ``post_processing`` chains effects, applied to every
simulation or in the dataloader as training-time augmentation. Nothing here imports the
simulators; ``simulators.base`` imports ``post_processing``. This file holds no imports.
"""
