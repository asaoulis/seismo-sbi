"""Waveform processing: instrument response removal, tapering, bandpass, resampling.

These are pure transformations on obspy.Stream — no file I/O.
"""

import obspy
from obspy import Stream, Inventory

_DEFAULT_PREFILTER = dict(
    pre_filt=[0.005, 0.01, 0.1, 0.2],
    taper=True,
    taper_fraction=0.05,
)

_DEFAULT_FILTER = dict(freqmin=0.02, freqmax=0.05, corners=4, zerophase=False)


def deconvolve_and_filter(
    stream: Stream,
    inventory: Inventory = None,
    remove_response: bool = True,
    prefilter_kwargs: dict = None,
    filter_kwargs: dict = None,
    target_sr: float = None,
) -> Stream:
    """Process a raw waveform Stream into SBI-ready data.

    Steps: merge gaps → [remove instrument response] → cosine taper →
           bandpass filter → resample to target_sr.

    Args:
        stream: Raw input traces (any sampling rate).
        inventory: Required when remove_response=True.
        remove_response: Whether to deconvolve the instrument response.
        prefilter_kwargs: Overrides for remove_response pre-filter/taper
            (merged with defaults: pre_filt, taper, taper_fraction).
        filter_kwargs: Overrides for bandpass filter
            (merged with defaults: freqmin, freqmax, corners, zerophase).
        target_sr: Resample to this rate (Hz) after filtering.
            If None, no resampling is performed.

    Returns:
        Processed Stream (displacement if response removed, else counts).
    """
    pf_kw = {**_DEFAULT_PREFILTER, **(prefilter_kwargs or {})}
    filt_kw = {**_DEFAULT_FILTER, **(filter_kwargs or {})}

    st = stream.copy()
    st = st.merge(method=0, fill_value="latest")

    if remove_response:
        if inventory is None:
            raise ValueError("inventory must be provided when remove_response=True")
        st = st.remove_response(inventory=inventory, output="DISP", **pf_kw)

    st = st.taper(max_percentage=0.01, type="cosine")
    st = st.filter("bandpass", **filt_kw)

    if target_sr is not None:
        st = st.resample(target_sr)

    return st
