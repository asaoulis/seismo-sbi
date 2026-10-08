"""Statistical priors for dataset generation.

:class:`EventCatalogue` and :func:`load_catalogue` ingest a catalogue;
:func:`latlon_to_km_offsets` and :func:`km_offsets_to_latlon` convert to and from local
kilometres; the Gutenberg-Richter helpers fit a b-value and sample magnitudes;
:mod:`moment_tensor_sampling` draws uniform orientations at a fixed moment; :mod:`samplers` wraps all
of it into closures; and :mod:`parameter_sampler` draws every parameter from its configured sampler.
"""
