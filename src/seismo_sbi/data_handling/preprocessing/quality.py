"""Quality checks for preprocessed waveform windows.

These are pure predicates on obspy.Stream — no file I/O, no side effects.
"""

from datetime import timedelta
from typing import List, Optional, Tuple

import numpy as np
from obspy import Stream, UTCDateTime


def check_window_quality(
    stream: Stream,
    receivers: List[str],
    sampling_rate: float,
    duration: timedelta,
    min_completeness: float = 0.9,
    min_rms: float = 0.0,
) -> Tuple[bool, str]:
    """Check that a preprocessed window meets minimum quality requirements.

    Two checks are applied per station / component:
    1. **Completeness**: at least *min_completeness* fraction of the expected
       sample count must be present (catches gapped or truncated windows).
    2. **RMS floor**: the RMS of each trace must exceed *min_rms* (catches
       all-zero or constant-value traces, which indicate a dead channel).

    Args:
        stream: Preprocessed Stream (already at *sampling_rate*).
        receivers: Ordered list of station names to check.
        sampling_rate: Expected sampling rate (Hz).
        duration: Expected window length (timedelta).
        min_completeness: Minimum fraction of expected samples (0–1).
        min_rms: Minimum acceptable per-trace RMS value.

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
            n = len(tr.data)
            completeness = n / expected_samples
            if completeness < min_completeness:
                return False, (
                    f"{sta}.{tr.stats.channel}: completeness {completeness:.2f} "
                    f"< {min_completeness}"
                )
            rms = float(np.sqrt(np.mean(tr.data.astype(float) ** 2)))
            if rms <= min_rms:
                return False, (
                    f"{sta}.{tr.stats.channel}: RMS {rms:.3g} "
                    f"<= floor {min_rms}"
                )
        stations_ok += 1

    if stations_ok == 0:
        return False, "no recognised stations found in stream"

    return True, ""
