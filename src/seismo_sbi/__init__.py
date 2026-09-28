"""Simulation-based inference of earthquake sources from seismograms.

``simulators`` turn source parameters into seismograms; ``sbi`` holds the pipeline, noise models,
compression and training; ``evaluation`` validates trained models; ``moment_tensor`` holds the
moment-tensor conventions and source-type physics; ``priors``, ``data_quality`` and
``data_handling`` prepare the inputs; ``plotting`` and ``utils`` serve all of them. Import the
module you need: package files hold only docstrings.
"""
