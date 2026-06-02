"""Catalogue-driven statistical priors for SBI dataset generation.

Public API:
  * :class:`EventCatalogue`, :func:`load_catalogue` - catalogue ingestion.
  * :func:`latlon_to_km_offsets`, :func:`km_offsets_to_latlon` - local geo conversion.
  * Gutenberg-Richter helpers (:func:`fit_b_value_aki`, :func:`magnitude_to_m0`,
    :class:`GutenbergRichterModel`, :func:`estimate_mc_maxcurvature`).
  * :func:`uniform_moment_tensor_on_sphere` - uniform MT orientation at fixed M0.
  * Sampler factories (:func:`make_catalogue_location_sampler`,
    :func:`make_gutenberg_richter_mt_sampler`) used by the dataset generator.
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
