"""Source-type lune figures of moment-tensor ensembles.

The colours and reference marker styles shared with the corner plot, the HPD contours and the
delta-percentile tensors of each ensemble, reference markers, beachballs moved into gutters beside
the lune, the legend and a zoomed inset. Each function draws on a frame from
:func:`~seismo_sbi.plotting.lune.plot_lune_frame`; a moment tensor is the 6-vector
``m_rr, m_tt, m_pp, m_rt, m_rp, m_tp``.
"""

import matplotlib.pyplot as plt
import numpy as np
import pyrocko.moment_tensor as mtm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from seismo_sbi.moment_tensor.conventions import create_matrix
from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta
from seismo_sbi.plotting.lune import kde_hpd_contour_levels, kde_on_grid, plot_kde_contours_on_lune
from seismo_sbi.plotting.rocko_beachball_patch import plot_beachball_on_axes

#: One colour per overlaid ensemble, in dict order, shared by the lune contours and the corner
#: plot. Hex codes, because the corner-plot library accepts only those or its own mapped names.
LUNE_ENSEMBLE_COLORS = ['#6495ED', '#FF0000', '#800080', '#008000', '#A52A2A',
                        '#FFA500', '#008080', '#FF00FF', '#808000', '#FFD700', '#00FFFF']

#: Scatter styles for reference solutions overlaid alongside the primary one, cycled in the
#: order the ``extra_references`` mapping gives them.
LUNE_REFERENCE_STYLES = [
    {"marker": "*", "color": "gold",        "s": 380},
    {"marker": "s", "color": "dodgerblue",  "s": 200},
    {"marker": "^", "color": "magenta",     "s": 230},
    {"marker": "P", "color": "darkorange",  "s": 230},
    {"marker": "D", "color": "limegreen",   "s": 170},
    {"marker": "X", "color": "red",         "s": 230},
]


def relocate_beachballs_outside_lune(ax, bm, specs, diameter=0.06, gutter_pad=1.3):
    """Draw each collected beachball in a gutter just outside the lune rather than on top
    of the scatter/KDE it annotates.

    Beachballs belonging to the same posterior (``spec['group']``) stay together on one side,
    stacked in their original vertical order with a minimal order-preserving nudge so they
    don't overlap each other. Each *group* is then assigned to a (side, column) slot: the
    lighter side of the inner column is preferred, the opposite side is the first fallback,
    and only if a group still collides in y does it move to a further-out column (larger ``|x|``).
    A thin leader line connects each relocated beachball back to its true location (kept marked
    by the caller). ``specs`` is a list of dicts with keys ``mt, x, y, color, edge`` and
    optional ``group`` (defaults to ``color``) / ``linewidth``.
    """
    if not specs:
        return

    # A beachball is circular in display, so its diameter is a fraction of the axes height;
    # offset the gutters by about one radius so the balls clear the frame at any delta.
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    half_w = 0.5 * diameter * bbox.height / bbox.width * (xmax - xmin)
    x_l, _ = bm(-30, 0)
    x_r, _ = bm(30, 0)
    min_gap = 1.05 * diameter * (ymax - ymin)   # beachball vertical extent + small margin

    # group beachballs by posterior so each ensemble's balls stay together as a unit
    groups, order = {}, []
    for s in specs:
        key = s.get('group', s['color'])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)

    # within each group: stack in y-order, push apart to min_gap, recentre on its own midpoint
    g_specs, g_ys, g_interval = {}, {}, {}
    for key in order:
        col = sorted(groups[key], key=lambda s: s['y'])
        ys = [s['y'] for s in col]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + min_gap)
        shift = 0.5 * (col[0]['y'] + col[-1]['y']) - 0.5 * (ys[0] + ys[-1])
        ys = [y + shift for y in ys]
        g_specs[key], g_ys[key] = col, ys
        g_interval[key] = (ys[0] - 0.5 * min_gap, ys[-1] + 0.5 * min_gap)

    # assign each group to a (side, column) slot, never overlapping another group's y-interval
    slots = {}                                   # (side, col) -> occupied y-intervals
    load = {'L': 0, 'R': 0}                       # balls per side, for balancing the inner column
    placement = {}
    for key in sorted(order, key=lambda k: -0.5 * (g_interval[k][0] + g_interval[k][1])):
        lo, hi = g_interval[key]
        chosen, col_idx = None, 0
        while chosen is None:
            for side in sorted(('L', 'R'), key=lambda sd: load[sd]):
                if all(hi < a or lo > b for a, b in slots.get((side, col_idx), [])):
                    chosen = (side, col_idx)
                    break
            col_idx += 1
        slots.setdefault(chosen, []).append((lo, hi))
        load[chosen[0]] += len(g_specs[key])
        placement[key] = chosen

    # draw each group at its slot's x; column index pushes outer columns further from the lune
    col_step = 2.3 * half_w
    for key in order:
        side, col_idx = placement[key]
        if side == 'L':
            gx = x_l - gutter_pad * half_w - col_idx * col_step
        else:
            gx = x_r + gutter_pad * half_w + col_idx * col_step
        for s, yb in zip(g_specs[key], g_ys[key]):
            ax.plot([s['x'], gx], [s['y'], yb], color='gray', lw=0.6, alpha=0.7, zorder=9)
            plot_beachball_on_axes(ax, s['mt'], gx, yb, diameter=diameter,
                                   color_t=s['color'], edgecolor=s['edge'],
                                   zorder=10, linewidth=s.get('linewidth', 1))


def add_lune_legend(ax, labels, colors, title=None, loc='upper left', fontsize=13,
                     extra_markers=None):
    """Add a per-ensemble colour legend to a lune plot (single source of truth for the
    colour->label mapping shared by plot_lunes / plot_lunes_kde and the evaluation wrappers).

    ``extra_markers`` (optional) is a list of ``{label, color, marker}`` dicts for additional
    reference-MT scatter overlays; each is appended as a marker handle below the line handles."""
    from matplotlib.lines import Line2D
    extra_markers = list(extra_markers or [])
    handles = [Line2D([0], [0], lw=2.2, color=colors[i % len(colors)], label=lab)
               for i, lab in enumerate(labels)]
    handles += [Line2D([0], [0], lw=0, marker=em["marker"], color=em["color"],
                       markeredgecolor='black', markersize=12, label=em["label"])
                for em in extra_markers]
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=fontsize, title=title, framealpha=0.9)


def delta_percentile_tensors(moment_tensors, delta_deg, quantiles):
    """The tensor of ``moment_tensors`` whose lune latitude is closest to each percentile of ``delta_deg``."""
    percentile_mts = []
    for q in quantiles:
        d_q = np.percentile(delta_deg, q)
        idx = np.argmin(np.abs(delta_deg - d_q))
        percentile_mts.append(moment_tensors[idx])
    return percentile_mts


def _mark_tensor(ax, bm, moment_tensor, **scatter_kwargs):
    """Scatter one tensor at its lune position; returns ``(x, y, pyrocko MomentTensor)`` for its beachball."""
    mt = np.array(moment_tensor)
    tg, td = mts6_to_gamma_delta(mt.reshape(1, -1))
    tx, ty = bm(tg, td)
    ax.scatter(tx, ty, **scatter_kwargs)
    return tx[0], ty[0], mtm.MomentTensor(m_up_south_east=create_matrix(mt))


def draw_lune_scatter(ax, bm, ensembles_mt, true_tensor, plot_beachballs):
    """Each ensemble of ``ensembles_mt`` (``{name: (n, 6)}``) as points on the lune. With
    ``plot_beachballs`` the true tensor and the 5th, 50th and 95th delta percentiles of the first
    ensemble are marked; returns their beachball specs for :func:`relocate_beachballs_outside_lune`."""
    colors = LUNE_ENSEMBLE_COLORS
    beachball_specs = []
    for i, samples_MT in enumerate(ensembles_mt.values()):
        gamma, delta = mts6_to_gamma_delta(samples_MT)
        x, y = bm(gamma, delta)
        ax.scatter(x, y, color=colors[i % len(colors)], alpha=0.3, s=6, marker='o')
        if i == 0 and plot_beachballs:
            percentile_mts = delta_percentile_tensors(samples_MT, delta, [5, 50, 95])
            for idx, mt in enumerate([true_tensor] + percentile_mts):
                if mt is None:
                    continue
                facecolor = 'black' if idx == 0 else colors[i % len(colors)]
                tx, ty, mt = _mark_tensor(ax, bm, mt, color=facecolor, alpha=1.0, marker='o', s=20, zorder=11)
                beachball_specs.append({'mt': mt, 'x': tx, 'y': ty,
                                        'color': facecolor, 'edge': 'black', 'linewidth': 0.5,
                                        'group': 'truth' if idx == 0 else i})
    return beachball_specs


def draw_lune_kde(ax, bm, ensembles_mt, true_tensor, plot_beachballs):
    """The 68 % (solid) and 95 % (dashed) HPD contours of each ensemble of ``ensembles_mt``, the true
    tensor as a diamond, and with ``plot_beachballs`` delta-percentile tensors of each ensemble.
    Returns ``(lune_angles, beachball_specs)``: ``(gamma_deg, delta_deg)`` per ensemble and the specs
    for :func:`relocate_beachballs_outside_lune`."""
    colors = LUNE_ENSEMBLE_COLORS
    # the quantiles cycle so that more than three overlaid ensembles still get percentile tensors
    qs = [[5,50,95], [50], [5,50,95]]
    lune_angles = []
    beachball_specs = []
    for i, samples_MT in enumerate(ensembles_mt.values()):
        g, d = mts6_to_gamma_delta(samples_MT)
        lune_angles.append((g, d))
        plot_kde_contours_on_lune(ax, bm, g, d, colors=colors[i % len(colors)],
                                  linestyles=('-', '--'), linewidths=(1.8, 1.5))
        percentile_mts = delta_percentile_tensors(samples_MT, d, qs[i % len(qs)])

        # truth drawn once (first ensemble, identical across ensembles); percentiles per ensemble
        mts_to_draw = ([(true_tensor, True)] if i == 0 else []) + [(m, False) for m in percentile_mts]
        for mt, is_true_mt in mts_to_draw:
            if mt is None:
                continue
            if not plot_beachballs and not is_true_mt:
                continue
            facecolor = 'peru' if is_true_mt else colors[i % len(colors)]
            marker = 'd' if is_true_mt else 'o'
            tx, ty, mt = _mark_tensor(ax, bm, mt, color=facecolor, alpha=1.0, marker=marker,
                                     s=320 if is_true_mt else 40, zorder=11)
            if plot_beachballs:
                beachball_specs.append({'mt': mt, 'x': tx, 'y': ty,
                                        'color': facecolor, 'edge': 'black', 'linewidth': 1,
                                        'group': 'truth' if is_true_mt else i})
    return lune_angles, beachball_specs


def draw_extra_references(ax, bm, reference_tensors):
    """Scatter each reference tensor of ``reference_tensors`` (``{label: 6-vector or None}``) with
    the next style of ``LUNE_REFERENCE_STYLES``; a None entry is skipped but uses up its style.
    Returns the ``{label, color, marker}`` legend specs."""
    specs = []
    for j, (label, ref_mt) in enumerate(reference_tensors.items()):
        if ref_mt is None:
            continue
        style = LUNE_REFERENCE_STYLES[j % len(LUNE_REFERENCE_STYLES)]
        ref_mt = np.array(ref_mt)
        rg, rd = mts6_to_gamma_delta(ref_mt.reshape(1, -1))
        rx, ry = bm(rg, rd)
        ax.scatter(rx, ry, color=style["color"], marker=style["marker"], s=style["s"],
                   edgecolors="black", linewidths=1.0, zorder=12)
        specs.append({"label": label, "color": style["color"], "marker": style["marker"]})
    return specs


def draw_primary_reference(ax, bm, moment_tensor, **scatter_kwargs):
    """Scatter the primary reference tensor at its lune position."""
    pr_mt = np.array(moment_tensor)
    pg, pd = mts6_to_gamma_delta(pr_mt.reshape(1, -1))
    px, py = bm(pg, pd)
    ax.scatter(px, py, **scatter_kwargs)


def add_lune_inset(ax, bm, lune_angles, true_tensor):
    """A zoomed inset of the HPD contours within 8 degrees of the true tensor, or of the first
    ensemble's mean lune angles without one; returns the inset axes."""
    colors = LUNE_ENSEMBLE_COLORS
    if true_tensor is not None:
        tg, td = mts6_to_gamma_delta(true_tensor.reshape(1, -1))
        tg = float(tg[0]); td = float(td[0])
    else:
        if len(lune_angles) > 0:
            tg = float(np.mean(lune_angles[0][0]))
            td = float(np.mean(lune_angles[0][1]))
        else:
            tg, td = 0.0, 0.0

    gmin, gmax = max(-30, tg - 8.0), min(30, tg + 8.0)
    dmin, dmax = max(-90, td - 8.0), min(90, td + 8.0)

    iax = inset_axes(ax, width="25%", height="40%", loc="upper right", borderpad=0.8)
    iax.set_in_layout(True)
    iax.set_zorder(ax.get_zorder()+1)
    iax.set_facecolor("white")

    gx_i = np.linspace(gmin, gmax, 160)
    gy_i = np.linspace(dmin, dmax, 240)
    GX_i, GY_i = np.meshgrid(gx_i, gy_i)
    XX_i, YY_i = bm(GX_i, GY_i)

    for i, (g, d) in enumerate(lune_angles):
        _, _, Z_i, _ = kde_on_grid(g, d, gx_i, gy_i)
        thr68, thr95 = kde_hpd_contour_levels(Z_i, levels=(0.6827, 0.9545))
        iax.contour(XX_i, YY_i, Z_i, levels=[thr95, thr68], colors=colors[i % len(colors)],
                    linestyles=['--', '-'], linewidths=[1.2, 1.5])

    xcorn, ycorn = bm([gmin, gmax, gmin, gmax], [dmin, dmin, dmax, dmax])
    iax.set_xlim(min(xcorn), max(xcorn))
    iax.set_ylim(min(ycorn), max(ycorn))

    center_g = 0.5 * (gmin + gmax)
    center_d = 0.5 * (dmin + dmax)
    xtick_vals = np.linspace(gmin, gmax, 5)
    ytick_vals = np.linspace(dmin, dmax, 5)
    xtick_pos, _ = bm(xtick_vals, np.full_like(xtick_vals, center_d))
    _, ytick_pos = bm(np.full_like(ytick_vals, center_g), ytick_vals)
    iax.set_xticks(xtick_pos)
    iax.set_xticklabels([f"{v:.0f}" for v in xtick_vals])
    iax.set_yticks(ytick_pos)
    iax.set_yticklabels([f"{v:.0f}" for v in ytick_vals])
    iax.set_xlabel(r"$\gamma$ (°)", fontsize=18)
    iax.set_ylabel(r"$\delta$ (°)", fontsize=18)
    tx, ty = bm(tg, td)
    iax.scatter(tx, ty, color='black', alpha=1.0, marker='x', s=120)
    return iax
