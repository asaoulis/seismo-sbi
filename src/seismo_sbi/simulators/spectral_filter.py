"""A Butterworth filter designed at one sampling rate, applied to a trace sampled at another.

:func:`filter_and_shift` designs the filter obspy's ``Stream.filter`` would build at the observed
data's rate, evaluates its complex response at the trace's own FFT frequencies, and applies it
together with a sub-sample time shift as one product in the frequency domain. The FFT is
zero-padded past the filter's decay time, so neither the filter tail nor the shift wraps around.
"""

import functools

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq
from scipy.signal import iirfilter, sosfreqz, zpk2sos

#: Impulse-response decay, relative to its start, that the zero-padding must reach.
TAIL_TOLERANCE = 1e-9


def butterworth_sos(filter_kwargs, sampling_rate_hz):
    """Second-order sections of obspy's ``bandpass``, ``lowpass`` or ``highpass`` at ``sampling_rate_hz``."""
    nyquist_hz = 0.5 * sampling_rate_hz
    corners = filter_kwargs.get("corners", 4)
    if filter_kwargs["type"] == "bandpass":
        if filter_kwargs["freqmax"] >= nyquist_hz:
            raise ValueError("filter freqmax is at or above the filter rate's Nyquist frequency")
        corner_freqs = [filter_kwargs["freqmin"] / nyquist_hz, filter_kwargs["freqmax"] / nyquist_hz]
        btype = "band"
    elif filter_kwargs["type"] in ("lowpass", "highpass"):
        corner_freqs = filter_kwargs["freq"] / nyquist_hz
        btype = filter_kwargs["type"]
    else:
        raise ValueError(f"unsupported filter type {filter_kwargs['type']!r}")
    zeros, poles, gain = iirfilter(corners, corner_freqs, btype=btype, ftype="butter", output="zpk")
    return zpk2sos(zeros, poles, gain)


@functools.lru_cache(maxsize=64)
def _response(filter_items, filter_sampling_rate_hz, n_fft, dt_s):
    """The filter's response at the ``n_fft``-point FFT frequencies of a trace sampled every ``dt_s``."""
    filter_kwargs = dict(filter_items)
    sos = butterworth_sos(filter_kwargs, filter_sampling_rate_hz)
    freqs_hz = rfftfreq(n_fft, dt_s)
    response = np.zeros(len(freqs_hz), dtype=complex)
    below_nyquist = freqs_hz < 0.5 * filter_sampling_rate_hz
    response[below_nyquist] = sosfreqz(sos, worN=freqs_hz[below_nyquist], fs=filter_sampling_rate_hz)[1]
    return response


@functools.lru_cache(maxsize=64)
def filter_tail_s(filter_items, filter_sampling_rate_hz):
    """Time (s) for the filter's slowest pole to decay to ``TAIL_TOLERANCE``."""
    sos = butterworth_sos(dict(filter_items), filter_sampling_rate_hz)
    poles = np.concatenate([np.roots(section[3:]) for section in sos])
    return np.log(TAIL_TOLERANCE) / np.log(np.max(np.abs(poles))) / filter_sampling_rate_hz


def filter_and_shift(data, dt_s, filter_kwargs, filter_sampling_rate_hz, shift_s=0.0, upsampling=1):
    """``data`` filtered as obspy would at ``filter_sampling_rate_hz``, read ``shift_s`` later.

    ``data`` is ``(n_samples,)`` sampled every ``dt_s``; ``filter_kwargs`` are those of
    ``Stream.filter``. The result is ``(upsampling * n_samples,)``: sample ``m`` is the filtered
    trace at ``m * dt_s / upsampling + shift_s`` after the first input sample, with
    ``0 <= shift_s < dt_s``. Content above half the filter rate is removed.
    """
    filter_items = tuple(sorted(filter_kwargs.items()))
    tail_samples = int(np.ceil(filter_tail_s(filter_items, filter_sampling_rate_hz) / dt_s))
    n_samples, n_fft = len(data), next_fast_len(len(data) + tail_samples + 1, real=True)
    response = _response(filter_items, float(filter_sampling_rate_hz), n_fft, float(dt_s))
    spectrum = rfft(np.asarray(data, dtype=float), n_fft) * response
    if filter_kwargs.get("zerophase", False):
        forward = irfft(spectrum, n_fft)[:n_samples]
        backward = irfft(rfft(forward[::-1], n_fft) * response, n_fft)[:n_samples]
        spectrum = rfft(backward[::-1], n_fft)
    spectrum *= np.exp(2j * np.pi * rfftfreq(n_fft, dt_s) * shift_s)
    return upsampling * irfft(spectrum, upsampling * n_fft)[:upsampling * n_samples]
