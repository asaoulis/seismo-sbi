"""Data quality control: compare a reference synthetic with observed waveforms and decide, per
station and component, whether to keep, time-shift or drop it.

The metrics, policies, alignment and serialisation take numpy arrays and small trace
descriptors; the caller reads the files and runs the forward model.
"""
