import obspy
from obspy.clients.fdsn.mass_downloader import RectangularDomain, \
    Restrictions, MassDownloader, CircularDomain
import os


# Rectangular domain containing parts of southern Germany.
#idomain = RectangularDomain(minlatitude=38.2, maxlatitude=39.15,
#                           minlongitude=-29, maxlongitude=-27)
domain = CircularDomain(latitude=38.67, longitude=-28.15, minradius=0, maxradius=3.00)
# domain = CircularDomain(latitude=32.370, longitude=-16.7000, minradius=0, maxradius=5.00)
restrictions = Restrictions(
    station="*",
    starttime=obspy.UTCDateTime(2021, 11, 1),
    endtime=obspy.UTCDateTime(2021, 11, 11),
    # starttime=obspy.UTCDateTime(2021, 9, 1),
    # endtime=obspy.UTCDateTime(2021, 9, 2),
    chunklength_in_sec=86400,
    channel_priorities= ["HH[ZNE12]", "BH[ZNE12]"],
    reject_channels_with_gaps=False,
    minimum_length=0.0,
    minimum_interstation_distance_in_m=100.0)

stor = ("stor/{station}/{starttime.year}.{starttime.julday:03g}/{network}.{station}.{location}.{channel}.{starttime.year}.{starttime.julday:03g}.mseed")
def get_mseed_storage(network, station, location, channel, starttime, endtime):
    ROOT = "/data/UPFLOW/_azores_obs_/IPMA_DATA_EARLY/stor/"
    return os.path.join(ROOT, "%s/%s.%03g/%s.%s.%s.%s.%s.%03g.mseed" % (station, starttime.year, starttime.julday, network, station, location, channel, starttime.year, starttime.julday))

mseed_storage = get_mseed_storage

mdl = MassDownloader(providers=["http://ceida.ipma.pt"])
mdl.download(domain, restrictions, mseed_storage=mseed_storage,
             stationxml_storage="stations")