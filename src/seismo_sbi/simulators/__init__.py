"""Forward models: what turns source parameters into seismograms.

``base.Simulator`` is the interface; ``sources`` and ``receivers`` carry what goes in and where
it is recorded; ``post_processing`` holds the nuisance effects applied to every simulation;
``registry`` maps a configuration's ``simulation_type`` to a builder. One subpackage per backend
(``instaseis``, ``cps``, ``axisem``). Import the module you need: this file holds no imports, so
a backend never drags in another backend's dependencies.
"""
