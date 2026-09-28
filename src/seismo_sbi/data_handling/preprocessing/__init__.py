"""Standard-seismology data preprocessing for seismo-sbi.

All functions operate on obspy.Stream / obspy.Inventory. ``io`` reads and writes waveforms,
``processing`` removes the response and filters, ``windowing`` cuts event and noise windows,
``quality`` checks a window, ``catalogue`` builds the event and noise catalogues, ``daily``
the day files and ``prepare_event`` one event file from a configuration block; HDF5 conversion
is confined to ``sbi_export.export_to_sbi_h5``.
"""
