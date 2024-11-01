import argparse
from pathlib import Path
import obspy
from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader, CircularDomain
import sys

def get_arguments():
    parser = argparse.ArgumentParser(description="Download seismic data and station XML files.")
    parser.add_argument(
        '-o', '--output_dir', type=Path, default=Path("./data"),
        help="Directory to save downloaded data. Defaults to ./data"
    )
    return parser.parse_args()

def check_output_directory(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError(f"The path {path} is not a directory.")
    except Exception as e:
        print(f"Error with output directory: {e}")
        sys.exit(1)

def save_response_files(output_dir, inv):
    outputs = output_dir / 'RESP'
    outputs.mkdir(parents=True, exist_ok=True)
    for network in inv:
        for station in network:
            for channel in station:
                loc = channel.location_code
                cha = channel.code

                try:
                    selected_inv = inv.select(network=network.code, station=station.code, location=loc, channel=cha)
                except:
                    continue
                selected_inv.write(
                    str(output_dir / 'RESP' / f"{network.code}.{station.code}.{channel.location_code}.{channel.code}.RESP"),
                    format="STATIONXML"
                )

def get_mseed_storage(output_dir, network, station, location, channel, starttime, endtime):
    return str(output_dir / f"{station}/{starttime.year}.{starttime.julday:03g}/{network}.{station}.{location}.{channel}.{starttime.year}.{starttime.julday:03g}.mseed")

def main():
    args = get_arguments()
    output_dir = args.output_dir
    check_output_directory(output_dir)

    domain = CircularDomain(latitude=38.67, longitude=-28.15, minradius=0, maxradius=5.00)

    restrictions = Restrictions(
        station="*",
        starttime=obspy.UTCDateTime(2022, 1, 10),
        endtime=obspy.UTCDateTime(2022, 1, 14),
        chunklength_in_sec=86400,
        channel_priorities=["HH[ZNE12]", "BH[ZNE12]"],
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=100.0
    )

    mdl = MassDownloader(providers=["http://ceida.ipma.pt"])
    mdl.download(
        domain, restrictions,
        mseed_storage=lambda *args: get_mseed_storage(output_dir, *args),
        stationxml_storage=str(output_dir / "stations.xml")
    )
    inv = obspy.read_inventory(str(output_dir / "stations.xml"))
    save_response_files(output_dir, inv)


if __name__ == "__main__":
    main()
