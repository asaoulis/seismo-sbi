from obspy.geodetics.base import gps2dist_azimuth
from collections import namedtuple
import numpy as np

ReceiverStats = namedtuple('ReceiverStats', [
    'distance',         # float: epicentral distance in km
    'azimuth',          # float: from source to receiver
    'back_azimuth',     # float: from receiver to source
    't0',               # float: time offset for Green's function window
    'vred',             # float: reduction velocity (km/s), optional
    'window',           # float: time window in seconds
    'event_depth'       # float: source depth in km
])

def build_objstats(receivers, source, window, vred=0.0, t0=0.0):
    """
    Converts Receivers and GenericPointSource into a list of ReceiverStats
    suitable for GF computation.

    Parameters:
    - receivers: Receivers object
    - source: GenericPointSource object
    - window: float, window duration (s)
    - vred: float, reduction velocity (km/s), default 0.0
    - t0: float, reference time shift, default 0.0

    Returns:
    - List[ReceiverStats]
    """
    src_lat = source.source_location.latitude
    src_lon = source.source_location.longitude
    src_depth = source.source_location.depth

    objstats = []

    for receiver in receivers.iterate():
        rec_lat = receiver.latitude
        rec_lon = receiver.longitude

        dist_m, az, baz = gps2dist_azimuth(src_lat, src_lon, rec_lat, rec_lon)
        dist_km = dist_m / 1000.0

        stats = ReceiverStats(
            distance=dist_km,
            azimuth=az,
            back_azimuth=baz,
            t0=t0,
            vred=vred,
            window=window,
            event_depth=src_depth
        )

        objstats.append(stats)

    return objstats



def load_velocity_model(filepath):
    """
    Loads a 1D velocity model using NumPy's loadtxt.
    Skips comments and returns (6, N) array for write_Model96.
    """
    return np.loadtxt(filepath).T  # shape: (6, N)