
import sys
import os
import argparse
import shutil
import pickle
from pathlib import Path
import multiprocessing as mp
from functools import partial
import yaml



from cartopy import crs as ccrs

import numpy as np
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

def plot_perturbations(fiducial, perturbations, ax=None, title=None, add_legend=False):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator

    # Select only a subset of perturbations for clarity
    perturbations = perturbations[:50, :, :]

    # Helper: stair-step arrays from layered model
    def stairs_from_layers(depths, vals, top=0.0):
        depths = np.asarray(depths).astype(float).ravel()
        vals = np.asarray(vals).astype(float).ravel()
        edges = np.concatenate(([top], depths))
        y = np.repeat(edges, 2)[1:-1]
        x = np.repeat(vals, 2)
        return x, y

    # Fiducial model
    depth_f = np.cumsum(fiducial[:, 0])
    vp_f = fiducial[:, 1]
    vs_f = fiducial[:, 2]

    # Perturbations: (P, N, 4)
    depth_p = np.cumsum(perturbations[:, :, 0], axis=1)
    vp_p = perturbations[:, :, 1]
    vs_p = perturbations[:, :, 2]

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 5))

    # --- Plot perturbations (grey) ---
    P = perturbations.shape[0]
    for i in range(P):
        d_i = depth_p[i]

        x_vp, y_vp = stairs_from_layers(d_i, vp_p[i])
        x_vs, y_vs = stairs_from_layers(d_i, vs_p[i])

        ax.plot(x_vp, y_vp, color='0.6', alpha=0.3, lw=0.8)
        ax.plot(x_vs, y_vs, color='0.6', alpha=0.3, lw=0.8)

    # --- Plot fiducial ---
    x_vp_f, y_vp_f = stairs_from_layers(depth_f, vp_f)
    x_vs_f, y_vs_f = stairs_from_layers(depth_f, vs_f)

    ax.plot(x_vp_f, y_vp_f, color='tab:blue', lw=2.2, label='Vp (fiducial)')
    ax.plot(x_vs_f, y_vs_f, color='tab:red', lw=2.2, label='Vs (fiducial)')

    # Axes styling
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Depth (km)')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='out')

    # Depth direction: 0 at surface, depth increases downward
    all_depths = np.concatenate(
        [np.array([0.0]), depth_f.ravel(), depth_p.ravel()]
    )
    bottom_depth = np.nanmax(all_depths)
    ax.set_ylim(bottom_depth, 0.0)

    # X limits
    all_vels = np.concatenate([vp_f, vs_f, vp_p.ravel(), vs_p.ravel()])
    xmin, xmax = np.nanmin(all_vels), np.nanmax(all_vels)
    pad = 0.03 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

    if title is not None:
        ax.set_title(title)

    # Legend (only once if shared axes)
    if add_legend:
        proxy = [
            Line2D([0], [0], color='tab:blue', lw=2.2, label='Vp (fiducial)'),
            Line2D([0], [0], color='tab:red', lw=2.2, label='Vs (fiducial)'),
            Line2D([0], [0], color='0.6', lw=1.5, alpha=0.4, label='Perturbations'),
        ]
        ax.legend(handles=proxy, frameon=True, loc='best')

    return ax
