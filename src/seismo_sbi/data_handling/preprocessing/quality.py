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
        sta_traces = stream.select(station=sta)
        if len(sta_traces) == 0:
            return False, f"station {sta!r} has no traces"

        for tr in sta_traces:
            data = tr.data.astype(float)
            n = len(data)
            label = f"{sta}.{tr.stats.channel}"

            completeness = n / expected_samples
            if completeness < min_completeness:
                return False, (
                    f"{label}: completeness {completeness:.2f} "
                    f"< {min_completeness}"
                )

            if min_npts is not None and n < min_npts:
                return False, (
                    f"{label}: {n} samples < {min_npts} required for "
                    f"SBI contract length"
                )

            if not np.all(np.isfinite(data)):
                n_bad = int(np.sum(~np.isfinite(data)))
                return False, f"{label}: {n_bad} non-finite sample(s) (NaN/Inf)"

            if np.all(data == 0.0):
                return False, f"{label}: all samples are zero"

            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms <= min_rms:
                return False, (
                    f"{label}: RMS {rms:.3g} <= floor {min_rms}"
                )

            if n > 1:
                flat_frac = float(np.sum(np.diff(data) == 0.0)) / (n - 1)
                if flat_frac > max_flat_fraction:
                    return False, (
                        f"{label}: flat-period fraction {flat_frac:.3f} "
                        f"> {max_flat_fraction}"
                    )

        stations_ok += 1

    if stations_ok == 0:
        return False, "no recognised stations found in stream"

    return True, ""
