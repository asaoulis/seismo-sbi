import argparse
from pathlib import Path
import obspy
from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader, GlobalDomain
import sys
import numpy as np
from obspy import read_inventory


def get_arguments():
    parser = argparse.ArgumentParser(description="Download seismic data and station XML files.")
    parser.add_argument(
        '-o', '--output_dir', type=Path, default=Path("/data/alex/long_valley/test"),
        help="Directory to save downloaded data. Defaults to ./data"
    )
    # Added: configurable stations file path
    parser.add_argument(
        '--stations_file', type=Path,
        default=Path(__file__).resolve().parent / "configs" / "long_valley" / "stations.txt",
        help="Path to stations.txt with columns: STATION NETWORK LAT LON"
    )
    # add providers list, default is all providers i.e. None
    parser.add_argument(
        '--providers', type=str, default=None,
        help="Comma-separated list of providers to query (e.g., 'IRIS,NCEDC'). Defaults to all providers."
    )
    # add datetime range for download
    parser.add_argument(
        '--starttime', type=str, default="1997-11-22T17:00:00",
        help="Start time for data download in ISO format (e.g., '1997-11-22T11:30:00')."
    )
    parser.add_argument(
        '--endtime', type=str, default="1997-11-22T17:30:00",
        help="End time for data download in ISO format (e.g., '1997-11-22T12:30:00')."
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

def get_mseed_storage(output_dir, network, station, location, channel, starttime, endtime):
    return str(output_dir / f"{station}/{starttime.year}.{starttime.julday:03g}/{network}.{station}.{location}.{channel}.{starttime.year}.{starttime.julday:03g}.mseed")

def load_station_codes(stations_path: Path):
    """
    Load (network, station) pairs using numpy from a file with columns:
    STATION NETWORK LAT LON
    """
    try:
        codes = np.loadtxt(stations_path, dtype=str, comments="#", usecols=(0, 1))
        # Ensure 2D array even for a single line
        if codes.ndim == 1:
            codes = codes.reshape(1, 2)
        # Return as (network, station)
        return [(net, sta) for sta, net in codes]
    except Exception as e:
        print(f"Error reading stations file {stations_path}: {e}")
        sys.exit(1)

def unpack_stationxml(output_dir: Path):
    """
    Unpacks downloaded StationXML files into individual RESP files for each channel.
    """
    stationxml_dir = output_dir / "stationxml"
    resp_dir = output_dir / "RESP"
    resp_dir.mkdir(exist_ok=True)

    print("Unpacking StationXML files to RESP...", flush=True)
    for xml_file in stationxml_dir.glob("*.xml"):
        try:
            inv = read_inventory(str(xml_file))
            for net in inv:
                for sta in net:
                    for cha in sta:
                        # Build RESP filename
                        resp_name = (
                            f"RESP.{net.code}."
                            f"{sta.code}."
                            f"{cha.location_code}."
                            f"{cha.code}"
                        )
                        out_file = resp_dir / resp_name

                        # Create a minimal inventory with only this channel
                        sub_inv = inv.select(
                            network=net.code,
                            station=sta.code,
                            location=cha.location_code,
                            channel=cha.code,
                        )

                        # Write RESP
                        sub_inv.write(str(out_file), format="STATIONXML")
                        print(f"Wrote {out_file}")
        except Exception as e:
            print(f"Could not process {xml_file}: {e}", flush=True)
            continue

def main():
    args = get_arguments()
    providers = args.providers.split(",") if args.providers else None
    output_dir = args.output_dir
    check_output_directory(output_dir)

    # Use all-Earth domain; station filtering is handled by Restrictions (network/station)
    domain = GlobalDomain()

    # Use the stations file provided via CLI
    stations_path = args.stations_file
    station_pairs = load_station_codes(stations_path)
    starttime = obspy.UTCDateTime(args.starttime)
    endtime = obspy.UTCDateTime(args.endtime)

    mdl = MassDownloader(providers=providers)
    for network, station in station_pairs:
        print(network, station, flush=True)
        try:
            restrictions = Restrictions(
                # Full day of 22/11/1997
                starttime=starttime,
                endtime=endtime,
                network=network,
                station=station,
                chunklength_in_sec=86400,
                channel_priorities=["BH[ZNE12]", "HH[ZNE12]"],
                reject_channels_with_gaps=False,
                minimum_length=0.0,
                minimum_interstation_distance_in_m=0
            )
            mdl.download(
                domain, restrictions,
                mseed_storage=lambda *args: get_mseed_storage(output_dir, *args),
                stationxml_storage=str(output_dir / "stationxml")
            )
        except Exception as e:
            print(f"Error downloading {network}.{station}: {e}", flush=True)
            continue
    
    unpack_stationxml(output_dir)
if __name__ == "__main__":
    main()
