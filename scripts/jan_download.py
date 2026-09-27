import argparse
from pathlib import Path
import obspy
from obspy.clients.fdsn.mass_downloader import Restrictions, MassDownloader, RectangularDomain
import sys


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Download JAN seismic data for a geographic box using MassDownloader."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=Path,
        default=Path("/data/alex/JAN"),
        help="Directory to save downloaded data. Defaults to ./JAN",
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


def save_response_files(output_dir: Path, inv: obspy.Inventory):
    """
    Write per-channel RESP-like StationXML files:

    RESP/NET.STA.LOC.CHA.RESP
    """
    outputs = output_dir / "RESP"
    outputs.mkdir(parents=True, exist_ok=True)

    for network in inv:
        for station in network:
            for channel in station:
                loc = channel.location_code
                cha = channel.code

                try:
                    selected_inv = inv.select(
                        network=network.code,
                        station=station.code,
                        location=loc,
                        channel=cha,
                    )
                except Exception:
                    continue

                selected_inv.write(
                    str(outputs / f"{network.code}.{station.code}.{loc}.{cha}.RESP"),
                    format="STATIONXML",
                )


def get_mseed_storage(output_dir: Path, network, station, location, channel, starttime, endtime):
    # <output_dir>/<STATION>/YEAR.JJJ/NET.STA.LOC.CHA.YEAR.JJJ.mseed
    return str(
        output_dir
        / f"{station}/{starttime.year}.{starttime.julday:03g}/{network}.{station}.{location}.{channel}.{starttime.year}.{starttime.julday:03g}.mseed"
    )


def main():
    args = get_arguments()
    output_dir = args.output_dir
    check_output_directory(output_dir)

    # Time window: from 2026-01-14 00:00:00 to now (UTC)
    starttime = obspy.UTCDateTime(2026, 1, 14, 0, 0, 0)
    endtime = obspy.UTCDateTime()

    # Geographic box restriction (RectangularDomain)
    # north: 38.2524, east: 49.3945, south: 22.7559, west: 21.123
    # domain = RectangularDomain(
    #     minlatitude=22.7559,
    #     maxlatitude=38.2524,
    #     minlongitude=21.123,
    #     maxlongitude=49.3945,
    # )
    # circular domain Event latitude [°]: 31.09 Event longitude [°]: 35.28
    # 5 degrees
    from obspy.clients.fdsn.mass_downloader import CircularDomain
    domain = CircularDomain(
        latitude=31.09,
        longitude=35.28,
        minradius=0.0,
        maxradius=5.0,
    )
    # Automatically discover all available stations; all providers
    mdl = MassDownloader()  # no providers argument => all supported providers

    # Restrictions: no explicit network/station, automatic discovery in domain
    restrictions = Restrictions(
        starttime=starttime,
        endtime=endtime,
        chunklength_in_sec=86400,  # daily chunks
        channel_priorities=["BH[ZNE12]", "HH[ZNE12]"],
        reject_channels_with_gaps=False,
        minimum_length=0.0,
        minimum_interstation_distance_in_m=0,
    )

    stationxml_dir = output_dir / "stationxml"

    try:
        mdl.download(
            domain,
            restrictions,
            mseed_storage=lambda *args: get_mseed_storage(output_dir, *args),
            stationxml_storage=str(stationxml_dir),
        )
    except Exception as e:
        print(f"Error during JAN download: {e}", flush=True)
        return

    # After download, read all StationXML and generate RESP files
    try:
        inv = None
        for xml_file in stationxml_dir.glob("*.xml"):
            this_inv = obspy.read_inventory(str(xml_file))
            if inv is None:
                inv = this_inv
            else:
                inv += this_inv

        if inv is not None:
            save_response_files(output_dir, inv)
        else:
            print(f"No StationXML files found in {stationxml_dir}", flush=True)
    except Exception as e:
        print(f"Error while creating RESP files: {e}", flush=True)


if __name__ == "__main__":
    main()