# rocko_beachball_patch.py
import numpy as num
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.transforms import IdentityTransform

from pyrocko.plot.beachball import *

# Import helpers from your old module (the user said to do this)
# e.g. from rocko_beachball import choose_transform, deco_part, eig2gx, project, NA, BeachballError, mtm
# We'll assume the user will do `from rocko_beachball import *` before using this file,
# but to be explicit you can also import the needed symbols here if preferred.

def plot_beachball_mpl(
        mt, axes,
        beachball_type='deviatoric',
        position=(0., 0.),
        size=None,
        zorder=0,
        color_t='red',
        color_p='white',
        edgecolor='black',
        linewidth=2,
        alpha=1.0,
        arcres=181,
        decimation=1,
        projection='lambert',
        size_units='points',
        view='top'):
    """
    Adapted: returns components needed for external placement and distortion correction.

    Returns:
        (collection, base_transform, position, size, raw_data)
    where:
        - collection: PatchCollection with polygons in unit-beachball coordinates,
                      transform set to IdentityTransform() (caller must set final transform)
        - base_transform: transform returned by choose_transform(axes, size_units, ...)
        - position: position array returned by choose_transform (useful metadata)
        - size: size scalar returned by choose_transform (useful metadata)
        - raw_data: original data list [(verts, facecolor, edgecolor, linewidth), ...]
    """
    # Resolve transform/position/size like original code
    transform, position, size = choose_transform(axes, size_units, position, size)

    # Apply decomposition / view selection
    mt = deco_part(mt, beachball_type, view)

    eig = mt.eigensystem()
    if eig[0] == 0. and eig[1] == 0. and eig[2] == 0:
        raise BeachballError('eigenvalues are zero')

    # Build raw data **in unit coordinates** (DO NOT multiply by size or add position).
    # Caller will perform placement/scaling using the returned `position` & `size`.
    data = []
    for (group, patches, patches_lower, patches_upper,
            lines, lines_lower, lines_upper) in eig2gx(eig, arcres):

        if group == 'P':
            color = color_p
        else:
            color = color_t

        # Note: project() returns coordinates in normalized unit-circle space
        for poly in patches_upper:
            verts = project(poly, projection)[:, ::-1]  # <-- unit coords, do NOT *size or +position
            if alpha == 1.0:
                data.append((verts[::decimation], color, color, linewidth))
            else:
                data.append((verts[::decimation], color, 'none', 0.0))

        for poly in lines_upper:
            verts = project(poly, projection)[:, ::-1]
            data.append((verts[::decimation], 'none', edgecolor, linewidth))

    # Build Polygon patches from unit coords (no baked-in translation/scale)
    patches = []
    for (path, facecolor, edgecol, lw) in data:
        patches.append(Polygon(xy=path, facecolor=facecolor,
                               edgecolor=edgecol, linewidth=lw, alpha=alpha))

    # Create PatchCollection but DO NOT give it a meaningful transform:
    # set to IdentityTransform so caller can set a composed transform later.
    collection = PatchCollection(patches, zorder=zorder, match_original=True)
    collection.set_transform(IdentityTransform())

    # DO NOT add to axes here. Caller will set transform and add_collection().
    # Return collection + metadata so caller can place it exactly where they want.
    return collection, transform, position, size, data


# rocko_beachball_helpers.py
import numpy as np
from matplotlib.transforms import Affine2D

def plot_beachball_on_axes(
        ax,
        mt,
        tx, ty,
        diameter=0.06,
        zorder=10,
        beachball_type="full",
        projection="lambert",
        view="top",
        color_t="red",
        color_p="white",
        edgecolor="black",
        linewidth=2,
        alpha=1.0,
        arcres=181):
    """
    Draw a beachball at (tx,ty) in DATA coordinates, correcting for axis
    distortion so that the beachball is always circular.

    Parameters
    ----------
    ax : matplotlib Axes
    mt : moment tensor
    tx, ty : float (data coordinates)
    diameter : float
        Diameter as FRACTION of the axes height (0.05–0.1 typical).
    zorder : int

    Returns
    -------
    collection : PatchCollection
    """

    # ------------------------------------------------------------------
    # 1) Create collection in UNIT beachball space (no scaling/position)
    # ------------------------------------------------------------------
    collection, base_transform, pos_meta, size_meta, raw_data = plot_beachball_mpl(
        mt, ax,
        beachball_type=beachball_type,
        position=(0., 0.),
        size=1.0,
        zorder=zorder,
        color_t=color_t,
        color_p=color_p,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        arcres=arcres,
        decimation=1,
        projection=projection,
        size_units="axes",
        view=view
    )

    # ------------------------------------------------------------------
    # 2) Convert target point (tx,ty) from DATA → AXES FRACTION coords
    # ------------------------------------------------------------------
    disp_x, disp_y = ax.transData.transform((tx, ty))
    ax_fx, ax_fy = ax.transAxes.inverted().transform((disp_x, disp_y))

    # ------------------------------------------------------------------
    # 3) Compute axis distortion: width vs height in pixels
    # ------------------------------------------------------------------
    ax.figure.canvas.draw()   # ensures bbox values are correct
    bbox = ax.get_window_extent()
    axes_w_px, axes_h_px = bbox.width, bbox.height
    scale_x = axes_h_px / axes_w_px   # <1 if axes wider than tall

    # ------------------------------------------------------------------
    # 4) Build scaling transform for unit beachball → desired diameter
    #
    #    diameter is fraction of axes height.
    #    For unit polygons, we multiply by (diameter / 2) in both axes.
    # ------------------------------------------------------------------
    r = diameter / 2.0

    scale_unit = (
        Affine2D()
        .translate(-ax_fx, -ax_fy)
        .scale(r, r)          # uniform: keeps beachball circular
        .translate(ax_fx, ax_fy)
    )

    # ------------------------------------------------------------------
    # 5) Distortion correction so circle stays circular
    # ------------------------------------------------------------------
    distortion_fix = (
        Affine2D()
        .translate(-ax_fx, -ax_fy)
        .scale(scale_x, 1.0)
        .translate(ax_fx, ax_fy)
    )

    # ------------------------------------------------------------------
    # 6) Compose final transform (then map to ax.transAxes)
    # ------------------------------------------------------------------
    final_transform = (scale_unit + distortion_fix) + ax.transAxes

    collection.set_transform(final_transform)
    collection.set_clip_on(False)
    collection.set_zorder(zorder)

    ax.add_collection(collection)
    return collection
