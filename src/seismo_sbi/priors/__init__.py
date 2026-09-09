"""Catalogue-driven statistical priors for dataset generation.

:class:`EventCatalogue` and :func:`load_catalogue` ingest a catalogue;
:func:`latlon_to_km_offsets` and :func:`km_offsets_to_latlon` convert to and from local
kilometres; the Gutenberg-Richter helpers fit a b-value and turn magnitude into scalar moment;
:mod:`moment_tensor` draws uniform orientations at a fixed moment; and :mod:`samplers` wraps all
of it into the closures the dataset generator calls.
"""
from .catalogue import EventCatalogue, load_catalogue
from .geo import km_offsets_to_latlon, latlon_to_km_offsets
from .gutenberg_richter import (
    GutenbergRichterModel,
    estimate_mc_maxcurvature,
    fit_b_value_aki,
    magnitude_to_m0,
)
from .moment_tensor import scalar_moment, uniform_moment_tensor_on_sphere
from .samplers import (
    make_catalogue_location_sampler,
    make_gutenberg_richter_mt_sampler,
)

__all__ = [
    "EventCatalogue",
    "load_catalogue",
    "latlon_to_km_offsets",
    "km_offsets_to_latlon",
    "GutenbergRichterModel",
    "estimate_mc_maxcurvature",
    "fit_b_value_aki",
    "magnitude_to_m0",
    "scalar_moment",
    "uniform_moment_tensor_on_sphere",
    "make_catalogue_location_sampler",
    "make_gutenberg_richter_mt_sampler",
]
