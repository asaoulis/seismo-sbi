"""Inversion of compressed data for the source parameters.

``likelihood`` samples a Gaussian likelihood with emcee; ``sbi_inference`` trains and samples a
neural posterior estimator on compressed simulations; ``least_squares`` iterates the score
compression to the maximum-likelihood source.
"""
