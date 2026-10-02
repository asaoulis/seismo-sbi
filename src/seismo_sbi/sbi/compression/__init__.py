"""Data compression to one summary per parameter.

``gaussian`` holds the score compressor and the wrapper around a trained network;
``derivative_stencil`` runs the finite-difference simulations the score needs. The neural
posterior estimator lives in :mod:`seismo_sbi.sbi.npe`.
"""
