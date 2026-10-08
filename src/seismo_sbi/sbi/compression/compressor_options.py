"""The options of each compressor in the ``compression`` block.

:func:`compressor_options` turns one entry of the block into the compressor's name and its options:
``optimal_score`` takes a noise covariance model and the noise windows it is estimated from,
``theory_optimal_score`` a data covariance, a noise level and a diagonal regularisation of the theory
covariance, and the multi-point and second-order compressors a white-noise level.
"""
from dataclasses import dataclass, fields
from typing import ClassVar

from seismo_sbi.utils.errors import InvalidConfiguration


@dataclass(frozen=True)
class OptimalScoreOptions:
    """``optimal_score``: the noise ``covariance`` model, estimated from the noise windows under ``path``."""

    covariance: str
    path: str = None
    compressor_type: ClassVar[str] = "optimal_score"


@dataclass(frozen=True)
class TheoryOptimalScoreOptions:
    """``theory_optimal_score``: the ``data_covariance`` model at ``noise_level`` (m; None takes each event's
    pre-event variances) plus the theory covariance, its diagonal regularised by
    ``diag_regularisation_magnitude`` times its largest variance."""

    data_covariance: str = None
    noise_level: float = None
    diag_regularisation_magnitude: float = 0.0
    compressor_type: ClassVar[str] = "theory_optimal_score"


@dataclass(frozen=True)
class MultiOptimalScoreOptions:
    """``multi_optimal_score``: white noise of standard deviation ``noise_level`` (m)."""

    noise_level: float
    compressor_type: ClassVar[str] = "multi_optimal_score"


@dataclass(frozen=True)
class SecondOrderScoreOptions:
    """``second_order_score``: white noise of standard deviation ``noise_level`` (m)."""

    noise_level: float
    compressor_type: ClassVar[str] = "second_order_score"


COMPRESSOR_OPTIONS = {options.compressor_type: options for options in (
    OptimalScoreOptions, TheoryOptimalScoreOptions, MultiOptimalScoreOptions, SecondOrderScoreOptions)}


def compressor_options(compressor_type, block):
    """``(name, options)`` of one ``compression`` entry of ``compressor_type``.

    An ``optimal_score`` block is ``{covariance: path}`` and is named ``optimal_score_<covariance>``;
    any other block holds its options as keys and is named by its type.
    """
    if compressor_type not in COMPRESSOR_OPTIONS:
        raise InvalidConfiguration(f"Invalid compression type {compressor_type}. Only "
                                   f"[ {', '.join(COMPRESSOR_OPTIONS)} ] allowed")
    if compressor_type == "optimal_score":
        if not isinstance(block, dict) or len(block) != 1:
            raise InvalidConfiguration(
                "optimal_score entries must be of the form:\n"
                "  - optimal_score:\n      <covariance_option>: <path>"
            )
        covariance, path = list(block.items())[0]
        return f"{compressor_type}_{covariance}", OptimalScoreOptions(covariance, path)
    options_class = COMPRESSOR_OPTIONS[compressor_type]
    block = block or {}
    unknown = sorted(set(block) - {field.name for field in fields(options_class)})
    if unknown:
        raise InvalidConfiguration(f"compression.{compressor_type}: unknown keys {unknown}; allowed: "
                                   f"{sorted(field.name for field in fields(options_class))}.")
    try:
        return compressor_type, options_class(**block)
    except TypeError as error:
        raise InvalidConfiguration(f"compression.{compressor_type}: {error}") from error
