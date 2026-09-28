"""Waveform processing: instrument response removal, rotation to north/east, bandpass, resampling.

These are pure transformations on obspy.Stream — no file I/O. Horizontal channels coded 1 and 2
are rotated to true north and east with the inventory azimuths; channels coded N and E are taken
at face value.
"""

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
    """Merge gaps, remove the instrument response, rotate 1/2 horizontals to north/east,
    cosine-taper, bandpass and resample a raw ``Stream``.

    :param stream: raw traces at any sampling rate.
    :param inventory: required when ``remove_response`` is True or any channel is coded 1 or 2.
    :param remove_response: whether to deconvolve the instrument response.
    :param prefilter_kwargs: overrides for the response-removal pre-filter and taper
        (``pre_filt``, ``taper``, ``taper_fraction``).
    :param filter_kwargs: overrides for the bandpass (``freqmin``, ``freqmax``, ``corners``,
        ``zerophase``).
    :param target_sr: resample to this rate in Hz after filtering; None keeps the rate.
    :returns: the processed ``Stream``, displacement if the response was removed, else counts.
    """
    pf_kw = {**_DEFAULT_PREFILTER, **(prefilter_kwargs or {})}
    filt_kw = {**_DEFAULT_FILTER, **(filter_kwargs or {})}

    st = stream.copy()
    st = st.merge(method=0, fill_value="latest")

    if remove_response:
        if inventory is None:
            raise ValueError("inventory must be provided when remove_response=True")
        st = st.remove_response(inventory=inventory, output="DISP", **pf_kw)
    st = rotate_horizontals_to_north_east(st, inventory)

    st = st.taper(max_percentage=0.01, type="cosine")
    st = st.filter("bandpass", **filt_kw)

    if target_sr is not None:
        st = st.resample(target_sr)

    return st


def rotate_horizontals_to_north_east(stream: Stream, inventory: Inventory = None) -> Stream:
    """``stream`` with every Z/1/2 channel set rotated to Z/N/E by the inventory azimuths and dips.

    Channels already coded N and E are left untouched. Raises ``ValueError`` when a 1 or 2
    channel is present and there is no inventory, or when a station lacks one of its Z, 1, 2.
    """
    if not any(trace.stats.channel[-1] in "12" for trace in stream):
        return stream
    if inventory is None:
        raise ValueError("an inventory is required to rotate channels coded 1/2 to north/east")
    stream = stream.rotate("->ZNE", inventory=inventory, components=("Z12",))
    unrotated = [trace.id for trace in stream if trace.stats.channel[-1] in "12"]
    if unrotated:
        raise ValueError(f"cannot rotate to north/east without a complete Z, 1, 2 set: {unrotated}")
    return stream
