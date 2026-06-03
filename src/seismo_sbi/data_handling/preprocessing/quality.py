"""Quality checks for preprocessed waveform windows.

These are pure predicates on obspy.Stream — no file I/O, no side effects.
"""

from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
from obspy import Stream, UTCDateTime

from seismo_sbi.instaseis_simulator.utils import compute_data_vector_length


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
    """Check that a preprocessed window meets minimum quality requirements.

    The following checks are applied per station / component:

    1. **Completeness**: at least *min_completeness* fraction of the expected
       sample count must be present (catches gapped or truncated windows).
    2. **Strict sample count** (when *min_npts* is set): the trace must have
       at least *min_npts* samples.  In the catalogue pipeline this is set to
       ``compute_data_vector_length(duration, sr) + 1`` (the SBI contract
       length) to prevent short arrays reaching the h5 export.
    3. **Non-finite values**: any NaN or Inf sample fails immediately.
    4. **All-zeros**: a trace whose every sample is exactly zero is rejected
       regardless of the *min_rms* setting.
    5. **RMS floor**: the RMS of each trace must exceed *min_rms*.
    6. **Flat-period fraction**: the fraction of consecutive identical samples
       must not exceed *max_flat_fraction*.  A high fraction indicates a dead
       or clipped channel, or a data gap filled with a constant.

    Args:
        stream: Preprocessed Stream (already at *sampling_rate*).
        receivers: Ordered list of station names to check.
        sampling_rate: Expected sampling rate (Hz).
        duration: Expected window length (timedelta).
        min_completeness: Minimum fraction of expected samples (0–1).
        min_rms: Minimum acceptable per-trace RMS value.
        max_flat_fraction: Maximum allowed fraction of consecutive identical
            samples in a trace (0–1).  Default 0.05 (5 %).
        min_npts: Minimum absolute sample count required per trace.  When set,
            enforced after the completeness check.  Pass
            ``compute_data_vector_length(duration_s, sr) + 1`` from the
            catalogue pipeline to guarantee the SBI contract array length.
            Default None (disabled).

    Returns:
        (ok: bool, reason: str)  — reason is empty string when ok=True.
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
    """Partition *receivers* into those passing quality and those dropped.

    Same per-station checks as :func:`check_window_quality`, but instead of
    failing the whole window on the first bad trace, each station is judged
    independently: a station is **kept** only if all its component traces pass,
    otherwise it is **dropped** (recording the failure reason).  This lets a
    single dead/flat/zero channel remove just that station rather than the entire
    event or noise window — essential when scaling to many stations, where one
    flaky station would otherwise reject most windows.

    Returns:
        (kept, dropped) where *kept* is the ordered list of station names that
        passed (a subset of *receivers*, order preserved) and *dropped* is a
        list of ``(station, reason)`` pairs for the rejected stations.
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
