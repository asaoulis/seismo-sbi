import obspy
from obspy.clients.fdsn import Client
import os


# Rectangular domain containing parts of southern Germany.
#idomain = RectangularDomain(minlatitude=38.2, maxlatitude=39.15,
#                           minlongitude=-29, maxlongitude=-27)



mdl = Client("http://ceida.ipma.pt/")
mdl.get_stations(latitude=38.67, longitude=-28.15, minradius=0, maxradius=3.00, channel="*H*", level="response", filename="stations.xml")

# print each station and lat lon
inv = obspy.read_inventory("stations.xml")

for network in inv:
    for station in network:
        print(station.code, station.latitude, station.longitude)