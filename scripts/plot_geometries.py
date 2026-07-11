import sys
import os
import argparse
import shutil
import pickle
from pathlib import Path
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
import yaml

from seismo_sbi.instaseis_simulator.receivers import Receivers


from cartopy import crs as ccrs

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from pyrocko import moment_tensor as mtm
from pyrocko.plot.beachball import plot_beachball_mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
from pyrocko import moment_tensor as mtm
from pyrocko.plot.beachball import plot_beachball_mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def convert_mt_convention(mt_rr_phi_theta):
    """(mnn, mee, mdd, mne, mnd, med)"""

    return [mt_rr_phi_theta[0], mt_rr_phi_theta[1], mt_rr_phi_theta[2], mt_rr_phi_theta[3], -mt_rr_phi_theta[4], -mt_rr_phi_theta[5]]



def create_matrix(moment_tensor_sol):
    moment_tensor_matrix = np.array([[moment_tensor_sol[0], moment_tensor_sol[3], moment_tensor_sol[4]],
                                        [moment_tensor_sol[3], moment_tensor_sol[1], moment_tensor_sol[5]],
                                        [moment_tensor_sol[4], moment_tensor_sol[5], moment_tensor_sol[2]]])
                                        
    return moment_tensor_matrix


def compute_mw(moment_tensor_matrix):
    """
    Compute moment magnitude Mw from full tensor (in N·m).
    Formula: Mw = (2/3) * log10(M0) - 6.0
    """
    M0 = np.sqrt(0.5 * np.sum(moment_tensor_matrix**2))
    Mw = (2.0 / 3.0) * (np.log10(M0) - 9.1)
    return Mw


def add_event_to_map(
    ax,
    event,
    projection=ccrs.PlateCarree(),
    beachball_type='full',
    beachball_size=20,   # points
    color_t='red',
    color_p='white',
    edgecolor='black',
    text_offset=(-0.3, 0.3),
    bb_offset=(0.0, 0.4),
    star_size=70,
):
    """
    Add a source marker (star), beachball, and Mw-labeled event name to a Cartopy map.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy axis to plot on.
    event : dict
        Dictionary with:
          - "moment_tensor": list[float] of 6 components
          - "source_location": [lat, lon, depth, 0]
          - "name": str
    projection : cartopy.crs
        Map projection (default PlateCarree).
    beachball_type : str
        'deviatoric', 'full', or 'dc'.
    beachball_size : int
        Diameter of beachball in points.
    text_offset : tuple
        (Δlon, Δlat) offset for text box from source.
    bb_offset : tuple
        (Δlon, Δlat) offset for beachball from source.
    star_size : int
        Marker size for hypocenter star.
    """

    lat, lon, depth, _ = event["source_location"]
    mt_sph = event["moment_tensor"]
    event_name = event["name"]

    # Convert and create MomentTensor
    mt_conv = convert_mt_convention(mt_sph)
    mt_matrix = create_matrix(mt_conv)
    mt = mtm.MomentTensor(m_up_south_east=mt_matrix)

    # Compute Mw
    Mw = compute_mw(mt_matrix)
    label = f"{event_name}\n$M_w$ {Mw:.2f}"

    # --- Plot actual source location ---
    ax.plot(
        lon, lat,
        marker='*',
        color='gold',
        markersize=10,
        markeredgecolor='black',
        transform=projection,
        zorder=12,
        label="Hypocenter"
    )

    # --- Plot beachball offset from source ---
    bb_lon = lon + bb_offset[0]
    bb_lat = lat + bb_offset[1]
    plot_beachball_mpl(
        mt,
        ax,
        beachball_type=beachball_type,
        position=(bb_lon, bb_lat),
        size=beachball_size,
        zorder=11,
        color_t=color_t,
        color_p=color_p,
        edgecolor=edgecolor,
        linewidth=1.2,
        alpha=1.0,
        projection='lambert',
        size_units='points',
        view='top'
    )

    # --- Label offset from beachball ---
    label_lon = bb_lon + text_offset[0]
    label_lat = bb_lat + text_offset[1]
    ax.text(
        label_lon,
        label_lat,
        label,
        transform=projection,
        fontsize=9,
        fontweight='bold',
        color='darkblue',
        ha='left',
        va='bottom',
        zorder=13,
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='white',
            edgecolor='darkblue',
            linewidth=0.6,
            alpha=0.8,
        ),
    )


config_paths = [Path('./configs/long_valley/synthetic_arrangement/LV2_synthetic_inversion.yaml'), Path('./configs/croatia/reduced_geom/event1_single.yaml'), Path('./configs/long_valley/LV2/LV_real.yaml'),  Path('./configs/japan/MIYTEC_synthetic_inversion.yaml'), ]
event_details = [
        {"moment_tensor": [2.5679503714262864e+16, 4088315540457834.0, 2.143170060059789e+16, 1138279029296896.2, -1.8092427075991144e+16, 1.5021729089952764e+16],
        "source_location": [37.636, -118.936, 5, 0],
        "name": "LV2",
        "offsets":((0.0, 0.4), (-0.3, 0.3))},
        # {"moment_tensor": [9.33217200877713e+16, -9.123066589872888e+16, -2091054189042384.5, -1.2547959671487482e+16, -1.6969891859945718e+16, 2.4214090206222812e+16],
        #  "source_location": [45.879, 16.028, 5, 0],
        #  "name": "Zagreb, 2020",
        #  "offsets":((0.0, 0.3), (-0.3, 0.2))},
        # {"moment_tensor": [2.5679503714262864e+16, 4088315540457834.0, 2.143170060059789e+16, 1138279029296896.2, -1.8092427075991144e+16, 1.5021729089952764e+16],
        # "source_location": [37.636, -118.936, 5, 0],
        # "name": "LV2",
        # "offsets":((0.0, 0.4), (-0.3, 0.3))},

        # {"moment_tensor": [1.0985209128388177e+18, -1.2581734406853128e+18, 2.229652527846495e+18, -8.380361649391023e+17, -3.985697520973377e+16, 4.0408647753592873e+18],
        # "source_location": [33.968, 139.414, 8, 0],
        # "name": "MIYTEC",
        # "offsets":((0.3, 0.), (0.25, -0.15))}
]

for config_path, event in zip(config_paths, event_details):
    print(f"Processing config: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)

    seismic_context_config = config.get("seismic_context", {})
    receivers_details = seismic_context_config.pop("stations_path")
    receiver_component_details = seismic_context_config.pop("station_components_path")
    receiver_time_shifts_details = seismic_context_config.pop("station_time_shifts_path", None)
    receivers = Receivers(receivers_details, receiver_component_details, receiver_time_shifts_details)

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    receivers.plot(ax=ax)
    bb, text = event["offsets"]
    # Add the focal mechanism and label
    add_event_to_map(ax, event, bb_offset=bb, text_offset=text)

    fig.savefig(f"plots/receiver_geometry_{config_path.stem}.png", dpi=300, bbox_inches='tight')
    print("Saved plot for config:", config_path)
    plt.close(fig)
