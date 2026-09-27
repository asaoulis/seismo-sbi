import argparse
from pathlib import Path
import obspy
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader, GlobalDomain
import sys
import numpy as np

# ---------------------------
# HARD-CODED CENTER LOCATION
# ---------------------------
CENTER_LAT = 45.88  
CENTER_LON = 16.028 


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Download seismic data and station XML files."
    )
    parser.add_argument(
        '-o', '--output_dir',
        type=Path,
        default=Path("/data/alex/croatia"),
        help="Directory to save downloaded data."
    )
    parser.add_argument(
        '--stations_file',
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "croatia" / "stations.txt",
        help="Path to stations.txt with columns: STATION NETWORK LAT LON"
    )

    # NEW OPTIONS
    parser.add_argument(
        '--fetch_stations_from_iris',
        action='store_true',
        help="Query IRIS for stations near a hard-coded lat/lon and write stations_file"
    )
    parser.add_argument(
        '--max_radius_deg',
        type=float,
        default=1.0,
        help="Maximum angular distance (degrees) from center lat/lon"
    )
    parser.add_argument(
        '--networks',
        type=str,
        default="*",
        help="Comma-separated network codes to include (default: all)"
    )

    return parser.parse_args()


def check_output_directory(path: Path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise NotADirectoryError(f"The path {path} is not a directory.")
    except Exception as e:
        print(f"Error with output directory: {e}")
        sys.exit(1)


def get_mseed_storage(output_dir, network, station, location, channel, starttime, endtime):
    return str(
        output_dir
        / f"{station}/{starttime.year}.{starttime.julday:03d}/"
          f"{network}.{station}.{location}.{channel}."
          f"{starttime.year}.{starttime.julday:03d}.mseed"
    )


def load_station_codes(stations_path: Path):
    """
    Load (network, station) pairs from a file with columns:
    STATION NETWORK LAT LON
    """
    try:
        codes = np.loadtxt(
            stations_path,
            dtype=str,
            comments="#",
            usecols=(0, 1)
        )
        if codes.ndim == 1:
            codes = codes.reshape(1, 2)
        return [(net, sta) for sta, net in codes]
    except Exception as e:
        print(f"Error reading stations file {stations_path}: {e}")
        sys.exit(1)


def fetch_and_write_stations_from_iris(
    stations_file: Path,
    max_radius_deg: float,
    networks: str,
    STATION_LIST = None,
    NETWORK_LIST = None,
    starttime = None,
    endtime = None
):
    """
    Query IRIS for stations within max_radius_deg of CENTER_LAT/LON
    and write stations_file in format:
    STATION NETWORK LAT LON
    """
    print(
        f"Querying IRIS for stations within {max_radius_deg}° of "
        f"({CENTER_LAT}, {CENTER_LON})",
        flush=True
    )
    providers = ["IRIS", "ODC", "GFZ", "INGV", "LMU"]#, "LMU", "EIDA"]
    all_stations = []
    for provider in providers:
        client = Client(provider)

        inv = client.get_stations(
            latitude=CENTER_LAT,
            longitude=CENTER_LON,
            maxradius=max_radius_deg,
            network=networks,
            level="station",
            starttime=starttime,
            endtime=endtime
        )
        all_stations.append(inv)
        station_names = [f"{net.code}.{sta.code}" for net in inv for sta in net.stations]
        print(provider, station_names, flush=True)
    inv = sum(all_stations[1:], all_stations[0])  # Merge inventories
    stations_file.parent.mkdir(parents=True, exist_ok=True)

    with stations_file.open("w") as f:
        f.write("# STATION NETWORK LAT LON\n")
        for network in inv:
            for station in network.stations:
                if STATION_LIST is not None and station.code not in STATION_LIST:
                    continue
                if NETWORK_LIST is not None and network.code not in NETWORK_LIST:
                    continue
                f.write(
                    f"{station.code} "
                    f"{network.code} "
                    f"{station.latitude:.6f} "
                    f"{station.longitude:.6f}\n"
                )

    print(f"Wrote {stations_file}", flush=True)

STATION_LIST = ["CACV", "KRJB", "BLY", "A273A", "MOZS", "CEY", "A262A", "PLIT", "PERS", "MOSL", "KALN", "DUGI", "HVAR", "BRJN", "OZLJ", "MORI", "OZLJ", "MPLH"]
# STATION_LIST = ["MORI", "HVAR"]
# 2020-12-28
# STATION_LIST = ["PERS", "KOGS", "BLY", "MOZS"]
STATION_LIST = ["CACV", "KRJB", "BLY", "MOZS",  "PLIT", "PERS", "MOSL",  "OZLJ", "OZLJ","KOGS", "KALN", "VINV" , "PERS"]

NETWORK_LIST = ["CR", "Z3", "SL", "MN", "HN"]

def main():
    args = get_arguments()
    output_dir = args.output_dir
    check_output_directory(output_dir)
    # start_time = obspy.UTCDateTime(2020, 12, 28, 5, 0, 0)
    # end_time = obspy.UTCDateTime(2020, 12, 28, 6, 0, 0)
    start_time = obspy.UTCDateTime(2024, 12, 28, 5, 0, 0)
    end_time = obspy.UTCDateTime(2024, 12, 28, 6, 0, 0)
    # Optional station discovery step
    if args.fetch_stations_from_iris:
        fetch_and_write_stations_from_iris(
            stations_file=args.stations_file,
            max_radius_deg=args.max_radius_deg,
            networks=args.networks,
            STATION_LIST=STATION_LIST,
            NETWORK_LIST=NETWORK_LIST,
            starttime=start_time,
            endtime=end_time
        )

    # Load station list (generated or pre-existing)
    station_pairs = load_station_codes(args.stations_file)

    domain = GlobalDomain()
    # mdl = MassDownloader(providers=["IRIS", "ODC", "GFZ", "LMU", "EIDA"])
    mdl = MassDownloader()  # INGV added 2024-06
    for network, station in station_pairs:
        print(network, station, flush=True)
        try:
            restrictions = Restrictions(
                starttime=start_time,
                endtime=end_time,
                network=network,
                station=station,
                channel="BH?,HH?,EH?,HN?",
                reject_channels_with_gaps=False,
                minimum_length=0.0,  # <--- ADD THIS
            )
            mdl.download(
                domain,
                restrictions,
                mseed_storage=lambda *a: get_mseed_storage(output_dir, *a),
                stationxml_storage=str(output_dir / "stationxml")
            )
        except Exception as e:
            print(f"Error downloading {network}.{station}: {e}", flush=True)
            continue


if __name__ == "__main__":
    main()
