"""Posterior figures: corner plots, lunes, beachballs and compression diagnostics.

:class:`PosteriorPlotter` takes posterior samples in the scaled space and draws them in physical
units: ChainConsumer corner plots, source-type lunes (scatter and KDE contours, drawn with
:mod:`seismo_sbi.plotting.lune_figures`), sampled and projected beachballs, and the
compression-error panels.
:class:`MomentTensorReparametrised` re-expresses moment-tensor samples as Mw and lune angles.
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from seismo_sbi.sbi.types.parameter_labels import ParameterInformation
from .patched_chainconsumer import CustomChainConsumer as ChainConsumer
from obspy.imaging.beachball import beach
from pyrocko.plot import beachball as rocko_beachball
import pyrocko.moment_tensor as mtm
from seismo_sbi.moment_tensor.conventions import create_matrix
from seismo_sbi.moment_tensor.decomposition import get_MW_and_epsilon, mechanism_parameters
from contextlib import contextmanager
import logging
from seismo_sbi.moment_tensor.lune_angles import mts6_to_gamma_delta
from seismo_sbi.plotting.lune import plot_lune_frame
from seismo_sbi.plotting.lune_figures import (
    LUNE_ENSEMBLE_COLORS, LUNE_REFERENCE_STYLES, add_lune_inset, add_lune_legend, draw_extra_references,
    draw_lune_kde, draw_lune_scatter, draw_primary_reference, relocate_beachballs_outside_lune)


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
            samples, theta0 = self.convert_samples(samples, theta0, custom_processing)
            converted_chain_dict[name] = (theta0, samples, None)
        return converted_chain_dict


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
        """Each ensemble of ``inversion_data`` (``name: (theta0, samples, ...)``) as points on the lune,
        at most ``num_samples`` each, with the first ensemble's truth and delta-percentile beachballs."""
        fig, ax = plt.subplots(figsize=(14, 14))
        bm = plot_lune_frame(ax)

        ensembles_mt, true_theta0 = self._lune_ensembles(inversion_data, num_samples)
        beachball_specs = draw_lune_scatter(ax, bm, ensembles_mt, true_theta0, plot_beachballs)
        relocate_beachballs_outside_lune(ax, bm, beachball_specs)
        extra_specs = draw_extra_references(ax, bm, self._reference_tensors(extra_references or {}))
        # The truth is the black dot drawn with the beachballs; without it a primary_reference takes its place.
        ref_specs = []
        if reference_label:
            if not (true_theta0 is not None and plot_beachballs) and primary_reference is not None:
                _, pr = self.get_moment_tensors(np.empty((0, 6)),
                                                np.asarray(primary_reference, dtype=float))
                draw_primary_reference(ax, bm, pr, color='black', marker='o', s=60,
                                       edgecolors='black', zorder=12)
                ref_specs = [{"label": reference_label, "color": "black", "marker": "o"}]
            elif true_theta0 is not None and plot_beachballs:
                ref_specs = [{"label": reference_label, "color": "black", "marker": "o"}]
        if legend:
            add_lune_legend(ax, list(inversion_data.keys()), LUNE_ENSEMBLE_COLORS,
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
        fig.canvas.draw()
        ensembles_mt, true_theta0 = self._lune_ensembles(inversion_data, num_samples)
        lune_angles, beachball_specs = draw_lune_kde(ax, bm, ensembles_mt, true_theta0, plot_beachballs)

        relocate_beachballs_outside_lune(ax, bm, beachball_specs)
        extra_specs = draw_extra_references(ax, bm, self._reference_tensors(extra_references or {}))
        ref_specs = self._primary_reference_legend(
            ax, bm, true_theta0, reference_label, primary_reference)
        if legend:
            add_lune_legend(ax, list(inversion_data.keys()), LUNE_ENSEMBLE_COLORS, title=legend_title,
                            extra_markers=ref_specs + extra_specs)

        iax = add_lune_inset(ax, bm, lune_angles, true_theta0) if plot_inset else None

        if figsave is None:
            if show:
                plt.show()
        else:
            kwargs = {} if iax is None else {"bbox_extra_artists": [iax]}
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight", **kwargs)
            plt.close()

        if show:
            plt.close()

    def _lune_ensembles(self, inversion_data, num_samples):
        """``({name: moment tensors (n, 6)}, first ensemble's true tensor or None)`` from at most
        ``num_samples`` shuffled samples of each ensemble."""
        ensembles_mt = {}
        true_theta0 = None
        for i, (name, (theta0, samples, *_)) in enumerate(inversion_data.items()):
            np.random.shuffle(samples)
            samples = samples[:num_samples]
            samples_MT, theta0_mt = self.get_moment_tensors(samples, theta0)
            if i == 0:
                true_theta0 = theta0_mt
            ensembles_mt[name] = samples_MT
        return ensembles_mt, true_theta0

    def _reference_tensors(self, extra_references):
        """``{label: moment tensor or None}`` of reference vectors given in the samples' units."""
        return {label: None if rmt is None
                else np.array(self.get_moment_tensors(np.empty((0, 6)), np.asarray(rmt, dtype=float))[1])
                for label, rmt in extra_references.items()}

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
            draw_primary_reference(ax, bm, pr_mt, color='peru', marker='d', s=320,
                                   edgecolors='black', linewidths=1.0, zorder=12)
            truth_drawn = True
        if truth_drawn and reference_label:
            return [{"label": reference_label, "color": "peru", "marker": "d"}]
        return []


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
