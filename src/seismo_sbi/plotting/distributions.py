"""Posterior figures: corner plots, lunes, beachballs and compression diagnostics.

:class:`PosteriorPlotter` takes posterior samples in the scaled space and draws them in physical
units: ChainConsumer corner plots, source-type lunes (scatter and KDE contours), sampled and
projected beachballs, and the compression-error panels.
:class:`MomentTensorReparametrised` re-expresses moment-tensor samples as Mw and lune angles.
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from seismo_sbi.sbi.types.parameter_labels import ParameterInformation
from .patched_chainconsumer import CustomChainConsumer as ChainConsumer
from obspy.imaging.beachball import beach
from pyrocko.plot import beachball as rocko_beachball
import pyrocko.moment_tensor as mtm
from seismo_sbi.moment_tensor.conventions import create_matrix
from seismo_sbi.moment_tensor.decomposition import get_MW_and_epsilon, mechanism_parameters
from .rocko_beachball_patch import plot_beachball_on_axes
from contextlib import contextmanager
import logging
from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta
from seismo_sbi.plotting.lune import plot_lune_frame, kde_on_grid, kde_hpd_contour_levels


# The angle distributions of a reparametrised moment tensor cover a periodic sample space,
# which the corner-plot library warns about on every call.
@contextmanager
def warning_logging_disabled(highest_level=logging.WARNING):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.

    :param highest_level: the maximum logging level in use.
        This would only need to be changed if a custom level greater than WARNING
        is defined.
    """

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


class DummyDataScaler:

    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_

    def inverse_transform(self, data):
        return data
    
    def transform(self, data):
        return data

class MomentTensorReparametrised:

                        
    parameters_info = [
                        ParameterInformation("$\gamma$", "°"),
                        ParameterInformation("$\delta$", '°'),
                        ParameterInformation("$M_w$", ""),
                        ParameterInformation(r"$\textrm{strike}$", '°'),
                        ParameterInformation(r"$\textrm{dip}$", '°'),
                        ParameterInformation(r"$\textrm{rake}$", '°'),
                        ]
    
    def __init__(self, data_scaler=None, parameters=None):
        """``parameters`` maps a sample vector to simulation inputs; None means each sample is the six
        moment-tensor components (N m) in the order m_rr, m_tt, m_pp, m_rt, m_rp, m_tp."""
        self.data_scaler = data_scaler
        self.parameters = parameters
        # no need for an extra scaling step
        dummy_scaler = DummyDataScaler(5)
        self.chain_plotter = PosteriorPlotter(dummy_scaler, self.parameters_info)

    def convert_samples(self, samples, theta0, custom_select):
        """``(converted samples (n, 6), converted theta0 (6,) or None)``: each vector as
        ``gamma_deg, delta_deg, Mw, strike_deg, dip_deg, rake_deg``
        (:func:`~seismo_sbi.moment_tensor.decomposition.mechanism_parameters`), the
        nodal plane the first one or the one ``custom_select`` picks from the pair."""
        moment_tensors = np.array([self._moment_tensor(sample) for sample in samples])
        converted = mechanism_parameters(moment_tensors, nodal_plane_choice=custom_select or None)
        if theta0 is not None:
            theta0_converted = mechanism_parameters(np.asarray(self._moment_tensor(theta0))[None],
                                                    nodal_plane_choice=custom_select or None)[0]
        else:
            theta0_converted = None
        return converted, theta0_converted

    def _moment_tensor(self, vector):
        if self.parameters is None:
            return vector
        return self.parameters.vector_to_simulation_inputs(vector, only_theta_fiducial=True)["moment_tensor"]

    def plot_chain_consumer(self, samples_theta0_dict, custom_processing = None, *args, extra_references=None, **kwargs):
        """Corner plot of the ensembles reparametrised as lune angles, Mw and one nodal plane.
        ``extra_references`` ``{label: vector}``, in the samples' units, are converted like a sample
        (the same nodal-plane choice) and drawn as markers. Returns the figure."""
        converted_chain_dict = self.convert_chain_dict(samples_theta0_dict, custom_processing)
        converted_references = {label: self.convert_samples(np.asarray(vector, dtype=float)[None], None, custom_processing)[0][0]
                                for label, vector in (extra_references or {}).items()}

        with warning_logging_disabled():
            return self.chain_plotter.plot_chain_consumer(converted_chain_dict, *args,
                                                          extra_references=converted_references, **kwargs)

    def convert_chain_dict(self, samples_theta0_dict, custom_processing):
        converted_chain_dict = {}
        for name, (theta0, samples, data_scaler, _) in samples_theta0_dict.items():
            if data_scaler is None:
                data_scaler = self.data_scaler
            if theta0 is not None:
                pass
            samples, theta0 = self.convert_samples(samples, theta0, custom_processing)
            converted_chain_dict[name] = (theta0, samples, None)
        return converted_chain_dict


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


def _relocate_beachballs_outside_lune(ax, bm, specs, diameter=0.06, gutter_pad=1.3):
    """Draw each collected beachball in a gutter just outside the lune rather than on top
    of the scatter/KDE it annotates.

    Beachballs belonging to the same posterior (``spec['group']``) stay together on one side,
    stacked in their original vertical order with a minimal order-preserving nudge so they
    don't overlap each other. Each *group* is then assigned to a (side, column) slot: the
    lighter side of the inner column is preferred, the opposite side is the first fallback,
    and only if a group still collides in y does it move to a further-out column (larger |x|).
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


def _add_lune_legend(ax, labels, colors, title=None, loc='upper left', fontsize=13,
                     extra_markers=None):
    """Add a per-ensemble colour legend to a lune plot (single source of truth for the
    colour->label mapping shared by plot_lunes / plot_lunes_kde and the evaluation wrappers).

    ``extra_markers`` (optional) is a list of ``{label, color, marker}`` dicts for additional
    reference-MT scatter overlays; each is appended as a marker handle below the line handles."""
    from matplotlib.lines import Line2D
    extra_markers = list(extra_markers or [])
    if plt.rcParams.get('text.usetex', False):
        # '%' is a LaTeX comment char; ChainConsumer can leave text.usetex on before we run.
        esc = lambda t: t.replace('%', r'\%') if isinstance(t, str) else t
        title = esc(title)
        labels = [esc(lab) for lab in labels]
        extra_markers = [{**em, "label": esc(em["label"])} for em in extra_markers]
    handles = [Line2D([0], [0], lw=2.2, color=colors[i % len(colors)], label=lab)
               for i, lab in enumerate(labels)]
    handles += [Line2D([0], [0], lw=0, marker=em["marker"], color=em["color"],
                       markeredgecolor='black', markersize=12, label=em["label"])
                for em in extra_markers]
    if handles:
        ax.legend(handles=handles, loc=loc, fontsize=fontsize, title=title, framealpha=0.9)


class PosteriorPlotter:

    def __init__(self, data_scaler, parameters_info : List[ParameterInformation], parameters = None, num_jobs = 0):

        self.data_scaler = data_scaler
        self.parameters_info = parameters_info
        self.parameters = parameters
        self.num_dim = len(parameters_info)
        self.num_jobs = num_jobs

    def plot_compression_errors(self, compressed_dataset, compressed_estimate = None, figname=None):

        ground_truths = compressed_dataset[:, :self.num_dim]
        compressions = compressed_dataset[:, self.num_dim:]
        ground_truths = self._transform_to_plotting_units(ground_truths)
        compressions = self._transform_to_plotting_units(compressions)
        if compressed_estimate is not None:
            compressed_estimate = self._transform_to_plotting_units(compressed_estimate.reshape(1, -1)).flatten()
            
        num_params = len(self.parameters_info)
        num_cols, _ = divmod(num_params, 2)
        num_cols +=1  # Assuming a 2x2 grid for each parameter

        fig, axes = plt.subplots(nrows=2, ncols=num_cols, figsize=(4*num_cols, 8))

        for i, (parameter, ax) in enumerate(zip(self.parameters_info, axes.ravel())):
            if i >= num_params:
                break  # Stop plotting if there are fewer subplots than parameters
            try:
                param_type = self.data_scaler.index_to_param_type[i]
            except AttributeError:
                param_type = None
            param_ground_truths = ground_truths[:, i] 
            param_compressions = compressions[:, i]
            ax.set_title(f"{parameter.name}")
            ax.scatter(param_ground_truths, param_compressions, label="Compression", marker='x', alpha=0.5)
            if compressed_estimate is not None:
                ax.hlines([compressed_estimate[i]], xmin=np.min(param_ground_truths), xmax = np.max(param_ground_truths), color="red", linestyle='--', label="Compression")
            # sort the data for plotting vs itself
            param_ground_truths = np.sort(param_ground_truths)
            ax.plot(param_ground_truths, param_ground_truths, label="Ground truth", color="green", linestyle='--')
            
            if param_type is not None and param_type == "moment_tensor" and len(self.parameters.bounds['moment_tensor']) == 1:
                ax.set_yscale('symlog')
                ax.set_xscale('symlog')
            
            unit_string = f"({parameter.unit})" if parameter.unit != "" else ""
            ax.set_xlabel(f"Ground truth {unit_string}")
            ax.set_ylabel(f"Compression {unit_string}")
            
        plt.tight_layout()

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close()

    def plot_chain_consumer(self, inversion_data, kde=True, extents=None, inverse=False, figsave= None, tick_font_size=30,
                            extra_references=None, *args, **kwargs):
        """Corner plot of each ``name: (theta0, samples, ...)`` ensemble in ``inversion_data``, the first
        one's ``theta0`` as the truth lines. ``extra_references`` ``{label: vector}``, in the samples'
        units, are drawn as markers in every two-parameter panel with the lune's reference styles and
        listed in a legend. Returns the figure."""
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
        colors = LUNE_ENSEMBLE_COLORS

        scaled_data_dict = {name: self._prepare_data_for_plotting(*data) 
                                for name, data in inversion_data.items()}

        parameters_label = [f"{parameters_info.name} [{parameters_info.unit}]" if parameters_info.unit != "" 
                                    else parameters_info.name for parameters_info in self.parameters_info]
                            
        i = 0
        c_plot = ChainConsumer()
        shade_first = len(scaled_data_dict) < 3
        for name, (samples, theta0) in scaled_data_dict.items():
            if extents is not None:
                for j in range(len(extents)):
                    samples = samples[samples[:, j] > extents[j][0]]
                    samples = samples[samples[:, j] < extents[j][1]]

            if i == 0:
                truth = theta0
                shade = shade_first
            else:
                shade = False
            c_plot.add_chain(samples, parameters=parameters_label, color=colors[i % len(colors)], name=name, shade=shade, linewidth=2.5)
            i+=1
        c_plot.configure(kde=[kde for _ in range(len(inversion_data))], shade_alpha=0.7, max_ticks=3, diagonal_tick_labels=False, inverse=inverse, tick_font_size=tick_font_size, label_font_size=40, summary=False, usetex=True, bar_shade=True)
        c_plot.configure_truth(lw=2)
        scale = 2.8*self.num_dim

        references = self._chain_consumer_references(extra_references or {}, parameters_label)
        plot_extents = extents
        if references and extents is None:
            plot_extents = self._extents_holding_references(scaled_data_dict, references, parameters_label)
        fig = c_plot.plotter.plot(figsize=(scale,scale), truth=truth, legend=False, extents=plot_extents,
                                  references=[(location, style) for _, location, style in references])
        fig.align_labels() 
        if references:
            self._add_chain_consumer_legend(fig, list(scaled_data_dict), references)

        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight")
        plt.close()
        return fig

    def _chain_consumer_references(self, extra_references, parameters_label):
        """``(label, {parameter label: value}, scatter style)`` for each reference vector, in plotting units."""
        references = []
        for j, (label, vector) in enumerate(extra_references.items()):
            style = LUNE_REFERENCE_STYLES[j % len(LUNE_REFERENCE_STYLES)]
            location = self._transform_to_plotting_units(np.asarray(vector, dtype=float).reshape(1, -1))[0]
            references.append((label, dict(zip(parameters_label, location)),
                               {"marker": style["marker"], "color": style["color"], "s": 0.6 * style["s"]}))
        return references

    @staticmethod
    def _extents_holding_references(scaled_data_dict, references, parameters_label, margin=0.08):
        """Axis ranges spanning the 0.5-99.5 percentiles of every ensemble and every reference, widened by ``margin``."""
        extents = {}
        for k, label in enumerate(parameters_label):
            values = [np.percentile(samples[:, k], [0.5, 99.5]) for samples, _ in scaled_data_dict.values()]
            values += [[location[label]] for _, location, _ in references]
            low, high = min(np.min(v) for v in values), max(np.max(v) for v in values)
            extents[label] = (low - margin * (high - low), high + margin * (high - low))
        return extents

    @staticmethod
    def _add_chain_consumer_legend(fig, chain_names, references):
        handles = [plt.Line2D([], [], color=LUNE_ENSEMBLE_COLORS[i % len(LUNE_ENSEMBLE_COLORS)], lw=4)
                   for i in range(len(chain_names))]
        handles += [plt.Line2D([], [], ls="", marker=style["marker"], color=style["color"], markeredgecolor="black",
                               markersize=18) for _, _, style in references]
        labels = list(chain_names) + [label for label, _, _ in references]
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), fontsize=34, frameon=False)

    def plot_lunes(self, inversion_data, num_samples=250, plot_beachballs=True, figsave=None, legend=True, extra_references=None, reference_label=None, primary_reference=None):

        # Project ensembles onto the standard Tape & Tape lune (Hammer) and scatter
        fig, ax = plt.subplots(figsize=(14, 14))
        bm = plot_lune_frame(ax)

        colors = LUNE_ENSEMBLE_COLORS
        true_theta0 = None
        beachball_specs = []

        for i, (name, (theta0, samples, *_)) in enumerate(inversion_data.items()):
            np.random.shuffle(samples)
            samples = samples[:num_samples]
            samples_MT, theta0_mt = self.get_moment_tensors(samples, theta0)
            if i == 0:
                true_theta0 = theta0_mt

            gamma, delta = mts6_to_gamma_delta(samples_MT)
            x, y = bm(gamma, delta)
            ax.scatter(x, y, color=colors[i % len(colors)], alpha=0.3, s=6, marker='o')
            if i ==0 and plot_beachballs:
                # beachballs for the true MT + delta percentiles of the first ensemble
                true_mt = true_theta0
                percentile_mts = []
                for q in [5, 50, 95]:
                    d_q = np.percentile(delta, q)
                    # find closest sample to this delta
                    idx = np.argmin(np.abs(delta - d_q))
                    percentile_mts.append(samples_MT[idx])

                for idx, mt in enumerate([true_mt] + percentile_mts):
                    if mt is None:
                        continue
                    facecolor = 'black' if idx == 0 else colors[i % len(colors)]
                    mt = np.array(mt)
                    tg, td = mts6_to_gamma_delta(mt.reshape(1, -1))
                    tx, ty = bm(tg, td)
                    mt = mtm.MomentTensor(m_up_south_east=create_matrix(mt))
                    # mark the true location; the beachball itself is relocated outside the lune
                    ax.scatter(tx, ty, color=facecolor, alpha=1.0, marker='o', s=20, zorder=11)
                    beachball_specs.append({'mt': mt, 'x': tx[0], 'y': ty[0],
                                            'color': facecolor, 'edge': 'black', 'linewidth': 0.5,
                                            'group': 'truth' if idx == 0 else i})

        _relocate_beachballs_outside_lune(ax, bm, beachball_specs)
        extra_specs = self._scatter_extra_references(ax, bm, extra_references or {})
        # Primary 'truth' here is the black dot (drawn in the i==0 block when plot_beachballs);
        # if absent but a primary_reference is supplied, draw it so it is shown consistently.
        ref_specs = []
        if reference_label:
            if not (true_theta0 is not None and plot_beachballs) and primary_reference is not None:
                _, pr = self.get_moment_tensors(np.empty((0, 6)),
                                                np.asarray(primary_reference, dtype=float))
                pr = np.array(pr)
                pg, pd = mts6_to_gamma_delta(pr.reshape(1, -1))
                px, py = bm(pg, pd)
                ax.scatter(px, py, color='black', marker='o', s=60,
                           edgecolors='black', zorder=12)
                ref_specs = [{"label": reference_label, "color": "black", "marker": "o"}]
            elif true_theta0 is not None and plot_beachballs:
                ref_specs = [{"label": reference_label, "color": "black", "marker": "o"}]
        if legend:
            _add_lune_legend(ax, list(inversion_data.keys()), colors,
                             extra_markers=ref_specs + extra_specs)

        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight")
        plt.close()


    def plot_lunes_kde(self, inversion_data, num_samples=2500, plot_beachballs=True, plot_inset=False, figsave=None, ax=None, show=True, legend=True, legend_title='solid 68%, dashed 95% HPD', extra_references=None, reference_label=None, primary_reference=None):
        """Plot 68%/95% HPD KDE contours for each ensemble on the projected lune, with a zoomed inset around truth ±8°.

        ``extra_references`` (optional) maps ``label -> MT 6-vector`` (canonical
        ``[Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]``) for additional published reference solutions, drawn as
        distinct scatter markers (see ``LUNE_REFERENCE_STYLES``) with their own legend entries.
        ``reference_label`` (optional) gives the primary 'truth' (gold diamond) its own legend
        entry. ``primary_reference`` (optional) is a primary-reference MT 6-vector drawn as that
        gold diamond when the ensembles carry no ``theta0`` truth (e.g. the dropout lune)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 14))
        else:
            fig = ax.get_figure()
        
        bm = plot_lune_frame(ax)

        colors = LUNE_ENSEMBLE_COLORS
        qs = [[5,50,95], [50], [5,50,95]]
        gx = np.linspace(-30, 30, 200)
        gy = np.linspace(-90, 90, 300)
        GX, GY = np.meshgrid(gx, gy)
        XX, YY = bm(GX, GY)

        # Cache per-ensemble gamma/delta for reuse in inset and store truth from first ensemble
        gd_list = []
        true_theta0 = None
        beachball_specs = []

        fig.canvas.draw()
        for i, (name, (theta0, samples, *_)) in enumerate(inversion_data.items()):
            np.random.shuffle(samples)
            samples = samples[:num_samples]
            samples_MT, theta0_mt = self.get_moment_tensors(samples, theta0)
            if i == 0:
                true_theta0 = theta0_mt
            g, d = mts6_to_gamma_delta(samples_MT)
            gd_list.append((g, d))
            _, _, Z, _ = kde_on_grid(g, d, gx, gy)
            thr68, thr95 = kde_hpd_contour_levels(Z, levels=(0.6827, 0.9545))
            ax.contour(XX, YY, Z, levels=[thr95, thr68], colors=colors[i % len(colors)],
                       linestyles=['--', '-'], linewidths=[1.5, 1.8])

            # delta percentiles for this ensemble; qs cycles so >3 overlaid ensembles
            # (e.g. station-dropout comparisons) don't IndexError.
            percentile_mts = []
            for q in qs[i % len(qs)]:
                d_q = np.percentile(d, q)
                # find closest sample to this delta
                idx = np.argmin(np.abs(d - d_q))
                percentile_mts.append(samples_MT[idx])

            # truth drawn once (first ensemble, identical across ensembles); percentiles per ensemble
            mts_to_draw = ([(true_theta0, True)] if i == 0 else []) + [(m, False) for m in percentile_mts]
            for mt, is_true_mt in mts_to_draw:
                if mt is None:
                    continue

                # If beachballs are OFF, only plot the true MT (scatter only)
                if not plot_beachballs and not is_true_mt:
                    continue

                mt = np.array(mt)

                facecolor = 'peru' if is_true_mt else colors[i % len(colors)]
                marker = 'd' if is_true_mt else 'o'

                tg, td = mts6_to_gamma_delta(mt.reshape(1, -1))
                tx, ty = bm(tg, td)
                mt = mtm.MomentTensor(m_up_south_east=create_matrix(mt))

                # mark the true location; the beachball itself is relocated outside the lune
                ax.scatter(
                    tx,
                    ty,
                    color=facecolor,
                    alpha=1.0,
                    marker=marker,
                    s=320 if is_true_mt else 40,
                    zorder=11
                )
                if plot_beachballs:
                    beachball_specs.append({'mt': mt, 'x': tx[0], 'y': ty[0],
                                            'color': facecolor, 'edge': 'black', 'linewidth': 1,
                                            'group': 'truth' if is_true_mt else i})

        _relocate_beachballs_outside_lune(ax, bm, beachball_specs)
        extra_specs = self._scatter_extra_references(ax, bm, extra_references or {})
        ref_specs = self._primary_reference_legend(
            ax, bm, true_theta0, reference_label, primary_reference)
        if legend:
            _add_lune_legend(ax, list(inversion_data.keys()), colors, title=legend_title,
                             extra_markers=ref_specs + extra_specs)

        iax = None
        if plot_inset:
            # Add zoomed inset centered on truth ±8 degrees
            # Determine center (truth). If not provided, use mean of first ensemble
            if true_theta0 is not None:
                tg, td = mts6_to_gamma_delta(true_theta0.reshape(1, -1))
                tg = float(tg[0]); td = float(td[0])
            else:
                if len(gd_list) > 0:
                    tg = float(np.mean(gd_list[0][0]))
                    td = float(np.mean(gd_list[0][1]))
                else:
                    tg, td = 0.0, 0.0

            gmin, gmax = max(-30, tg - 8.0), min(30, tg + 8.0)
            dmin, dmax = max(-90, td - 8.0), min(90, td + 8.0)

            # Build inset axes
            iax = inset_axes(ax, width="25%", height="40%", loc="upper right", borderpad=0.8)
            iax.set_in_layout(True)           # ensure included in tight bbox
            iax.set_zorder(ax.get_zorder()+1) # draw on top
            iax.set_facecolor("white")        # optional: make inset visible over map

            # Compute projected grid for the inset region
            gx_i = np.linspace(gmin, gmax, 160)
            gy_i = np.linspace(dmin, dmax, 240)
            GX_i, GY_i = np.meshgrid(gx_i, gy_i)
            XX_i, YY_i = bm(GX_i, GY_i)

            # Plot KDE contours for each ensemble inside inset
            for i, (g, d) in enumerate(gd_list):
                _, _, Z_i, _ = kde_on_grid(g, d, gx_i, gy_i)
                thr68, thr95 = kde_hpd_contour_levels(Z_i, levels=(0.6827, 0.9545))
                iax.contour(XX_i, YY_i, Z_i, levels=[thr95, thr68], colors=colors[i % len(colors)],
                            linestyles=['--', '-'], linewidths=[1.2, 1.5])

            # Center the inset view on the projected bounds
            xcorn, ycorn = bm([gmin, gmax, gmin, gmax], [dmin, dmin, dmax, dmax])
            iax.set_xlim(min(xcorn), max(xcorn))
            iax.set_ylim(min(ycorn), max(ycorn))

            # Add ticks showing gamma (x) and delta (y) degrees
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
            iax.set_xlabel(r"$\\gamma$ (°)", fontsize=18)
            iax.set_ylabel(r"$\\delta$ (°)", fontsize=18)
            # Plot the truth marker in the inset
            tx, ty = bm(tg, td)
            iax.scatter(tx, ty, color='black', alpha=1.0, marker='x', s=120)

        if figsave is None:
            if show:
                plt.show()
        else:
            kwargs = {} if iax is None else {"bbox_extra_artists": [iax]}
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight", **kwargs)
            plt.close()

        if show:
            plt.close()

    def _primary_reference_legend(self, ax, bm, true_theta0, reference_label, primary_reference):
        """Return the legend spec ``[{label,color,marker}]`` for the primary reference, the
        'truth' diamond (a published solution, say). If the ensembles carried a ``theta0`` truth
        it is already drawn by the main loop and only its legend entry is returned; if they did
        not (the station-dropout lune, whose configs have ``theta0=None``) but a
        ``primary_reference`` MT 6-vector is supplied, draw it here as the same peru diamond so it
        appears consistently. Returns ``[]`` when there is nothing to label."""
        truth_drawn = true_theta0 is not None
        if not truth_drawn and primary_reference is not None:
            _, pr_mt = self.get_moment_tensors(np.empty((0, 6)),
                                               np.asarray(primary_reference, dtype=float))
            pr_mt = np.array(pr_mt)
            pg, pd = mts6_to_gamma_delta(pr_mt.reshape(1, -1))
            px, py = bm(pg, pd)
            ax.scatter(px, py, color='peru', marker='d', s=320,
                       edgecolors='black', linewidths=1.0, zorder=12)
            truth_drawn = True
        if truth_drawn and reference_label:
            return [{"label": reference_label, "color": "peru", "marker": "d"}]
        return []

    def _scatter_extra_references(self, ax, bm, extra_references):
        """Overlay additional published reference MTs as distinct scatter markers on the
        projected lune. ``extra_references`` maps ``label -> MT 6-vector`` in the canonical
        ``[Mrr,Mtt,Mpp,Mrt,Mrp,Mtp]`` (USE/RTP) convention; each is routed through
        ``get_moment_tensors`` like the primary truth, so its lune position is consistent with
        the gold 'truth' marker. The lune (γ,δ) is scale- and basis-invariant, so absolute
        units / deviatoric-vs-full do not matter here.
        Returns a list of ``{label, color, marker}`` legend specs."""
        specs = []
        for j, (label, rmt) in enumerate(extra_references.items()):
            if rmt is None:
                continue
            style = LUNE_REFERENCE_STYLES[j % len(LUNE_REFERENCE_STYLES)]
            _, ref_mt = self.get_moment_tensors(np.empty((0, 6)), np.asarray(rmt, dtype=float))
            ref_mt = np.array(ref_mt)
            rg, rd = mts6_to_gamma_delta(ref_mt.reshape(1, -1))
            rx, ry = bm(rg, rd)
            ax.scatter(rx, ry, color=style["color"], marker=style["marker"], s=style["s"],
                       edgecolors="black", linewidths=1.0, zorder=12)
            specs.append({"label": label, "color": style["color"], "marker": style["marker"]})
        return specs

    def get_moment_tensors(self, samples, theta0):
        sample_mts = []
        for sample in samples:
            inputs = self.parameters.vector_to_simulation_inputs(sample, only_theta_fiducial=True)
            sample = inputs["moment_tensor"]
            sample_mts.append(sample)
        sample_mts = np.array(sample_mts)
        
        if theta0 is not None:
            theta_inputs = self.parameters.vector_to_simulation_inputs(theta0, only_theta_fiducial=True)
            theta0 = theta_inputs["moment_tensor"]
        return sample_mts, theta0

    def _prepare_data_for_plotting(self, theta0, samples, data_scaler = None, *args, **kwargs):

        if data_scaler is None:
            data_scaler = self.data_scaler
        if theta0 is not None:
            pass
        plotting_units_samples = self._transform_to_plotting_units(samples)
        if theta0 is not None:
            plotting_units_theta_0 = self._transform_to_plotting_units(theta0.reshape(1, -1)).flatten()
        else:
            plotting_units_theta_0 = None
        return plotting_units_samples,plotting_units_theta_0

    def _transform_to_plotting_units(self, samples):
        plotting_units_samples = np.copy(samples)
        for i in range(samples.shape[1]):
            scaler = self.parameters_info[i].scaling_transform
            plotting_units_samples[:,i] = scaler(samples[:,i])
        return plotting_units_samples

    def plot_beachball_samples(self, inversion_data, plot_path : Path = None):
        theta0, samples, data_scaler, _ = inversion_data
        if data_scaler is None:
            data_scaler = self.data_scaler

        np.random.shuffle(samples)
        if plot_path is not None:
            filename = plot_path.stem
            figsave_1 = plot_path.with_name(f"{filename}_samples.png")
            figsave_2 = plot_path.with_name(f"{filename}_fuzzy.png")
        else:
            figsave_1 = None
            figsave_2 = None
        self.plot_seperate_beachballs(samples, theta0, figsave=figsave_1)
        self.plot_beachball_projection_samples(samples, figsave=figsave_2)

    def plot_seperate_beachballs(self, plotting_units_samples, plotting_units_theta_0, sample_color='b', figsave = None):
        with plt.rc_context({'font.size' : 8}):
            fig, axes = plt.subplots(5,5, figsize=(16,12))

            for i, ax in enumerate(axes[:, 1:].ravel()):
                sample = plotting_units_samples[i]
                sample_inputs = self.parameters.vector_to_simulation_inputs(sample, only_theta_fiducial=True)
                sample = sample_inputs["moment_tensor"]
                M0_and_epsilon = get_MW_and_epsilon(sample)
                self.add_beachball_plot(ax, "", sample, M0_and_epsilon, col=sample_color)

            for i, ax in enumerate(axes[:, :1].ravel()):
                ax.axis('off')
                if i == 2 and plotting_units_theta_0 is not None:
                    theta0_inputs = self.parameters.vector_to_simulation_inputs(plotting_units_theta_0, only_theta_fiducial=True)
                    theta0_mt = theta0_inputs["moment_tensor"]
                    M0_and_epsilon = get_MW_and_epsilon(theta0_mt)
                    self.add_beachball_plot(ax, "", theta0_mt, M0_and_epsilon, col='plum')

            plt.subplots_adjust(wspace=-0.7, hspace=0.45)

            if figsave is None:
                plt.show()
            else:
                fig.savefig(figsave, dpi=200, transparent=True)
            plt.close()

    def plot_beachball_projection_samples(self, plotting_units_samples, sample_color='cornflowerblue', alpha=0.1, figsave = None, dpi=200):
        samples = plotting_units_samples
        np.random.shuffle(samples)
        fig, ax = plt.subplots(figsize=(5, 5))

        # Plot each moment tensor in the ensemble
        warning = False
        for sample in samples[:500]:
            sample_inputs = self.parameters.vector_to_simulation_inputs(sample, only_theta_fiducial=True)
            mt = sample_inputs["moment_tensor"]
            # Use beach() to plot the full moment tensor in 1x6 format

            try:
                b = beach(mt, linewidth=0.2, width=1.5, facecolor=sample_color, edgecolor=sample_color, alpha=alpha, nofill=True)
                ax.add_collection(b)
            except:
                if not warning:
                    print("Warning: beachball plotting failed for some samples. This is likely due to a bug in obspy.imaging.beachball.")
                    warning = True
                pass
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        plt.axis('off')
        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave, dpi=dpi, transparent=True)
        plt.close()
    
    def add_beachball_plot(self, ax, name, moment_tensor_sol, M0_epsilon, col = 'b', add_text = True):
        mt = mtm.MomentTensor(m_up_south_east=create_matrix(moment_tensor_sol))
        if add_text:
            extra_text = f"\n $M_W=${M0_epsilon[0]:.3f},\n$\\epsilon= {M0_epsilon[1]:.2f}$"
        else:
            extra_text = ""
        ax.axis('off')
        ax.set_title(f"{name}{extra_text}")
        rocko_beachball.plot_beachball_mpl(mt, ax, beachball_type='full', linewidth=1.5, color_t=col, size=50)
        ax.set_aspect("equal")
        ax.set_xlim((-0.1, 0.1))  
        ax.set_ylim((-0.1, 0.1))


# Standalone plotting: trajectories on the lune.

def plot_lune_histories(histories, figsave=None, linewidth=2.0, mark_endpoints=True):
    """
    Plot trajectories of MT histories on the Tape & Tape lune (Hammer projection).

    Parameters
    ----------
    histories : dict
        Name -> sequence of MTs in 6-component form [Mxx, Myy, Mzz, Mxy, Mxz, Myz]. Each
        value can be an array of shape (T, 6) or an iterable of length T with 6-vectors.
    figsave : str, optional
        Path to save the figure; if None, shows the plot.
    linewidth : float
        Line width for the trajectory.
    mark_endpoints : bool
        If True, mark start (circle) and end (cross) points of each trajectory.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    bm = plot_lune_frame(ax)

    colors = ['cornflowerblue', 'red', 'purple', 'green', 'brown']

    for i, (name, seq) in enumerate(histories.items()):
        arr = np.asarray(seq)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] != 6:
            raise ValueError(f"History '{name}' must be of shape (T, 6), got {arr.shape}.")
        # Convert to gamma/delta and project
        g, d = mts6_to_gamma_delta(arr)
        x, y = bm(g, d)
        col = colors[i % len(colors)]
        ax.plot(x, y, color=col, lw=linewidth, alpha=0.95, label=name)
        # Add small diamonds for intermediate points (exclude endpoints)
        if len(x) > 2:
            ax.scatter(x[1:-1], y[1:-1], color=col, s=18, marker='D', alpha=0.8, zorder=3)
        if mark_endpoints and len(x) > 0:
            ax.scatter(x[0], y[0], color=col, s=50, marker='o', zorder=4)
            ax.scatter(x[-1], y[-1], color=col, s=70, marker='x', zorder=4)

    if len(histories) > 0:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=min(3, len(histories)), frameon=False)

    if figsave is None:
        plt.show()
    else:
        fig.savefig(figsave, dpi=200, transparent=True, bbox_inches='tight')
    plt.close()
