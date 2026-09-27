"""Quality checks for preprocessed waveform windows.

These are pure predicates on obspy.Stream — no file I/O, no side effects.
"""

from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
from obspy import Stream



def check_window_quality(
    stream: Stream,
    receivers: List[str],
    sampling_rate: float,
    duration: timedelta,
    min_completeness: float = 0.9,
    min_rms: float = 0.0,
    max_flat_fraction: float = 0.05,
    min_npts: Optional[int] = None,
) -> Tuple[bool, str]:
    """Whether a preprocessed window meets the minimum quality requirements; ``(ok, reason)``.

    Per station and component: at least ``min_completeness`` of the expected samples are present;
    at least ``min_npts`` samples when set (``compute_data_vector_length(duration, sr) + 1`` in the
    catalogue pipeline); no NaN or Inf; not all zeros; RMS above ``min_rms``; and at most
    ``max_flat_fraction`` of consecutive identical samples (a dead or clipped channel, or a gap
    filled with a constant).

    :param stream: preprocessed ``Stream`` at ``sampling_rate``.
    :param receivers: ordered station names to check.
    :param sampling_rate: expected sampling rate in Hz.
    :param duration: expected window length, a ``timedelta``.
    :param min_completeness: minimum fraction of the expected samples, 0-1.
    :param min_rms: minimum per-trace RMS.
    :param max_flat_fraction: maximum fraction of consecutive identical samples, 0-1 (default 0.05).
    :param min_npts: minimum sample count per trace; None disables the check.
    :returns: ``(ok, reason)``; ``reason`` is empty when ``ok``.
    """
    expected_samples = int(duration.total_seconds() * sampling_rate)
    if expected_samples == 0:
        return False, "duration is zero"

    stations_ok = 0
    for sta in receivers:
        reason = _station_quality_failure(
            stream, sta, expected_samples,
            min_completeness=min_completeness, min_rms=min_rms,
            max_flat_fraction=max_flat_fraction, min_npts=min_npts,
        )
        if reason is not None:
            return False, reason
        stations_ok += 1

    if stations_ok == 0:
        return False, "no recognised stations found in stream"

    return True, ""


def _station_quality_failure(
    stream: Stream,
    sta: str,
    expected_samples: int,
    min_completeness: float = 0.9,
    min_rms: float = 0.0,
    max_flat_fraction: float = 0.05,
    min_npts: Optional[int] = None,
) -> Optional[str]:
    """Return the first quality-failure reason for a single station, or None.

    Applies, in order, the same per-trace checks documented on
    :func:`check_window_quality` (completeness, strict sample count, non-finite,
    all-zeros, RMS floor, flat-period fraction).  This is the shared kernel used
    by both the all-or-nothing :func:`check_window_quality` and the
    drop-bad-keep-good :func:`partition_window_quality`.
    """
    sta_traces = stream.select(station=sta)
    if len(sta_traces) == 0:
        return f"station {sta!r} has no traces"

    for tr in sta_traces:
        data = tr.data.astype(float)
        n = len(data)
        label = f"{sta}.{tr.stats.channel}"

        completeness = n / expected_samples
        if completeness < min_completeness:
            return f"{label}: completeness {completeness:.2f} < {min_completeness}"

        if min_npts is not None and n < min_npts:
            return (
                f"{label}: {n} samples < {min_npts} required for "
                f"SBI contract length"
            )

        if not np.all(np.isfinite(data)):
            n_bad = int(np.sum(~np.isfinite(data)))
            return f"{label}: {n_bad} non-finite sample(s) (NaN/Inf)"

        if np.all(data == 0.0):
            return f"{label}: all samples are zero"

        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms <= min_rms:
            return f"{label}: RMS {rms:.3g} <= floor {min_rms}"

        if n > 1:
            flat_frac = float(np.sum(np.diff(data) == 0.0)) / (n - 1)
            if flat_frac > max_flat_fraction:
                return (
                    f"{label}: flat-period fraction {flat_frac:.3f} "
                    f"> {max_flat_fraction}"
                )

    return None


def partition_window_quality(
    stream: Stream,
    receivers: List[str],
    sampling_rate: float,
    duration: timedelta,
    min_completeness: float = 0.9,
    min_rms: float = 0.0,
    max_flat_fraction: float = 0.05,
    min_npts: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Partition ``receivers`` into the stations passing quality and those dropped.

    The per-station checks of :func:`check_window_quality`, judged station by station: a station is
    kept only if all its component traces pass, so one dead channel removes that station rather than
    the whole window.

    :returns: ``(kept, dropped)``: the ordered station names that passed, and ``(station, reason)``
        pairs for the rest.
    """
    expected_samples = int(duration.total_seconds() * sampling_rate)
    if expected_samples == 0:
        return [], [(sta, "duration is zero") for sta in receivers]

    kept: List[str] = []
    dropped: List[Tuple[str, str]] = []
    for sta in receivers:
        reason = _station_quality_failure(
            stream, sta, expected_samples,
            min_completeness=min_completeness, min_rms=min_rms,
            max_flat_fraction=max_flat_fraction, min_npts=min_npts,
        )
        if reason is None:
            kept.append(sta)
        else:
            dropped.append((sta, reason))
    return kept, dropped
