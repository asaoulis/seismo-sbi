"""Data quality utilities: compare a reference synthetic against observed waveforms
and decide, per station, whether to keep / time-shift / drop it.

The package is deliberately small and generic: the metric, policy, alignment and
serialization layers take plain numpy arrays + lightweight descriptors (never a
pipeline or an h5 file), so they work for any simulator/source and are trivially
unit-testable. Scripts do the I/O and forward modelling and hand arrays in.
"""
