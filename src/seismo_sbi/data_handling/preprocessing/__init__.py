"""Standard-seismology data preprocessing for seismo-sbi.

Public API — all functions operate on obspy.Stream / obspy.Inventory.
HDF5 conversion is confined to sbi_export.export_to_sbi_h5.
"""

from .io import load_waveforms, load_inventory, write_window, find_mseed_files
from .processing import deconvolve_and_filter
from .windowing import (
    slice_event_window,
    make_noise_windows,
    compute_event_arrival_windows,
    filter_events_by_distance,
)
from .sbi_export import export_to_sbi_h5
from .quality import check_window_quality
from .catalogue import build_event_catalogue, build_noise_catalogue
from .daily import process_daily_files

__all__ = [
    "load_waveforms",
    "load_inventory",
    "write_window",
    "find_mseed_files",
    "deconvolve_and_filter",
    "slice_event_window",
    "make_noise_windows",
    "compute_event_arrival_windows",
    "filter_events_by_distance",
    "export_to_sbi_h5",
    "check_window_quality",
    "build_event_catalogue",
    "build_noise_catalogue",
    "process_daily_files",
]
