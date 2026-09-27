"""Forward models: what turns source parameters into seismograms.

``base.Simulator`` is the interface; ``sources`` and ``receivers`` carry what goes in and where
it is recorded; every simulation runs through a ``seismo_sbi.nuisance_effects`` chain;
``registry`` maps a configuration's ``simulation_type`` to a builder. One subpackage per backend
(``instaseis``, ``cps``, ``axisem``). Import the module you need: this file holds no imports, so
a backend never drags in another backend's dependencies.
"""
