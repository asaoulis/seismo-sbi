""" Here we add some custom functionality to ChainConsumer
the only addition is an 'inverse' option, which flips the 
trade-off plot triangle across the diagonal, as well as
all the label positions and orientations"""

import numpy as np
from scipy.interpolate import interp1d
import re
import matplotlib.pyplot as plt
import logging
import matplotlib
from matplotlib.ticker import MaxNLocator, ScalarFormatter, LogLocator

from chainconsumer.helpers import get_smoothed_bins, get_grid_bins
from chainconsumer.plotter import Plotter
from chainconsumer.chainconsumer import ChainConsumer

def remove_square_brackets(text):
    # Use regular expression to find and remove anything between square brackets
    cleaned_text = re.sub(r'\[.*?\]', '', text)
    return cleaned_text

class CustomChainConsumer(ChainConsumer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.plotter = CustomPlotter(self)
        self.config['inverse'] = False

    def configure(self, *args, **kwargs):
        inverse = kwargs.pop('inverse') if 'inverse' in kwargs else False
        super().configure(*args, **kwargs)
        self.config['inverse'] = inverse

class CustomPlotter(Plotter):

    def __init__(self, *args):
        super().__init__(*args)

    def plot(
        self,
        figsize="GROW",
        parameters=None,
        chains=None,
        extents=None,
        filename=None,
        display=False,
        truth=None,
        legend=None,
        blind=None,
        watermark=None,
        log_scales=None,
    ):  # pragma: no cover
        """ Plot the chain!

        Parameters
        ----------
        figsize : str|tuple(float)|float, optional
            The figure size to generate. Accepts a regular two tuple of size in inches,
            or one of several key words. The default value of ``COLUMN`` creates a figure
            of appropriate size of insertion into an A4 LaTeX document in two-column mode.
            ``PAGE`` creates a full page width figure. ``GROW`` creates an image that
            scales with parameters (1.5 inches per parameter). String arguments are not
            case sensitive. If you pass a float, it will scale the default ``GROW`` by
            that amount, so ``2.0`` would result in a plot 3 inches per parameter.
        parameters : list[str]|int, optional
            If set, only creates a plot for those specific parameters (if list). If an
            integer is given, only plots the fist so many parameters.
        chains : int|str, list[str|int], optional
            Used to specify which chain to show if more than one chain is loaded in.
            Can be an integer, specifying the
            chain index, or a str, specifying the chain name.
        extents : list[tuple[float]] or dict[str], optional
            Extents are given as two-tuples. You can pass in a list the same size as
            parameters (or default parameters if you don't specify parameters),
            or as a dictionary.
        filename : str, optional
            If set, saves the figure to this location
        display : bool, optional
            If True, shows the figure using ``plt.show()``.
        truth : list[float] or dict[str], optional
            A list of truth values corresponding to parameters, or a dictionary of
            truth values indexed by key
        legend : bool, optional
            If true, creates a legend in your plot using the chain names.
        blind : bool|string|list[string], optional
            Whether to blind axes values. Can be set to `True` to blind all parameters,
            or can pass in a string (or list of strings) which specify the parameters to blind.
        watermark : str, optional
            A watermark to add to the figure
        log_scales : bool, list[bool] or dict[bool], optional
            Whether or not to use a log scale on any given axis. Can be a list of True/False, a list of param
            names to set to true, a dictionary of param names with true/false
            or just a bool (just `True` would set everything to log scales).

        Returns
        -------
        figure
            the matplotlib figure

        """

        chains, parameters, truth, extents, blind, log_scales = self._sanitise(
            chains, parameters, truth, extents, color_p=True, blind=blind, log_scales=log_scales
        )
        names = [chain.name for chain in chains]

        if legend is None:
            legend = len(chains) > 1

        # If no chains have names, don't plot the legend
        legend = legend and len([n for n in names if n]) > 0

        # Calculate cmap extents
        unique_color_params = list(set([c.config["color_params"] for c in chains if c.config["color_params"] is not None]))
        num_cax = len(unique_color_params)
        color_param_extents = {}
        for u in unique_color_params:
            umin, umax = np.inf, -np.inf
            for chain in chains:
                if chain.config["color_params"] == u:
                    data = chain.get_color_data()
                    if data is not None:
                        umin = min(umin, data.min())
                        umax = max(umax, data.max())
            color_param_extents[u] = (umin, umax)

        grow_size = 1.5
        if isinstance(figsize, float):
            grow_size *= figsize
            figsize = "GROW"

        if isinstance(figsize, str):
            if figsize.upper() == "COLUMN":
                figsize = (5 + (1 if num_cax > 0 else 0), 5)
            elif figsize.upper() == "PAGE":
                figsize = (10, 10)
            elif figsize.upper() == "GROW":
                figsize = (grow_size * len(parameters) + num_cax * 1.0, grow_size * len(parameters))
            else:
                raise ValueError("Unknown figure size %s" % figsize)
        elif isinstance(figsize, float):
            figsize = (figsize * grow_size * len(parameters), figsize * grow_size * len(parameters))

        plot_hists = self.parent.config["plot_hists"]
        inverse = self.parent.config['inverse']
        flip = len(parameters) == 2 and plot_hists and self.parent.config["flip"]

        fig, axes, params1, params2, extents = self._get_figure(
            parameters, chains=chains, figsize=figsize, flip=flip, inverse=inverse, external_extents=extents, blind=blind, log_scales=log_scales
        )
        label_font_size = self.parent.config["label_font_size"]

        axl = axes.ravel().tolist()
        summary = self.parent.config["summary"]
        summary=True
        if summary is None:
            summary = len(parameters) < 5 and len(self.parent.chains) == 1
        if len(chains) == 1:
            self._logger.debug("Plotting surfaces for chain of dimension %s" % (chains[0].chain.shape,))
        else:
            self._logger.debug("Plotting surfaces for %d chains" % len(chains))
        cbar_done = []

        chain_points = [c for c in chains if c.config["plot_point"]]
        num_chain_points = len(chain_points)
        if num_chain_points:
            subgroup_names = list(set([c.name for c in chain_points]))
            subgroups = [[c for c in chain_points if c.name == n] for n in subgroup_names]
            markers = [group[0].config["marker_style"] for group in subgroups]  # Only one marker per group
            marker_sizes = [[g.config["marker_size"] for g in group] for group in subgroups]  # But size can diff
            marker_alphas = [group[0].config["marker_alpha"] for group in subgroups]  # Only one marker per group
        for i, p1 in enumerate(params1):
            for j, p2 in enumerate(params2):
                if i < j and not inverse:
                    continue
                if j < i and inverse:
                    continue
                ax = axes[i, j]
                do_flip = flip and i == len(params1) - 1

                # Plot the histograms
                if plot_hists and i == j:
                    if do_flip:
                        self._add_truth(ax, truth, p1)
                    else:
                        self._add_truth(ax, truth, None, py=p2)
                    max_val = None

                    # Plot each chain
                    for chain_idx, chain in enumerate(chains):
                        if p1 not in chain.parameters:
                            continue
                        if not chain.config["plot_contour"]:
                            continue

                        param_summary = summary and p1 not in blind
                        param_summary = (chain_idx ==0)
                        m = self._plot_bars(ax, p1, chain, flip=do_flip, inverse=inverse, summary=param_summary)

                        if max_val is None or m > max_val:
                            max_val = m

                    if num_chain_points and self.parent.config["global_point"]:
                        m = self._plot_point_histogram(ax, subgroups, p1, flip=do_flip)
                        if max_val is None or m > max_val:
                            max_val = m

                    if max_val is not None:
                        if do_flip:
                            ax.set_xlim(0, 1.1 * max_val)
                        else:
                            ax.set_ylim(0, 1.1 * max_val)

                else:
                    for chain in chains:
                        if p1 not in chain.parameters or p2 not in chain.parameters:
                            continue
                        if not chain.config["plot_contour"] or chain.config["show_as_1d_prior"]:
                            continue
                        h = None
                        if p1 in chain.parameters and p2 in chain.parameters:
                            h = self._plot_contour(ax, chain, p1, p2, color_extents=color_param_extents)
                        cp = chain.config["color_params"]
                        if h is not None and cp is not None and cp not in cbar_done:
                            cbar_done.append(cp)
                            aspect = figsize[1] / 0.15
                            fraction = 0.85 / figsize[0]
                            cbar = fig.colorbar(h, ax=axl, aspect=aspect, pad=0.03, fraction=fraction, drawedges=False)
                            label = cp
                            if label == "weights":
                                label = "Weights"
                            elif label == "log_weights":
                                label = "log(Weights)"
                            elif label == "posterior":
                                label = "log(Posterior)"
                            cbar.set_label(label, fontsize=label_font_size)
                            cbar.solids.set(alpha=1)

                    if num_chain_points:
                        self._plot_points(ax, subgroups, markers, marker_sizes, marker_alphas, p1, p2)

                    self._add_truth(ax, truth, p1, py=p2)

        colors = [c.config["color"] for c in chains]
        plot_points = [c.config["plot_point"] for c in chains]
        plot_contours = [c.config["plot_contour"] for c in chains]
        linestyles = [c.config["linestyle"] for c in chains]
        linewidths = [c.config["linewidth"] for c in chains]
        marker_styles = [c.config["marker_style"] for c in chains]
        marker_sizes = [c.config["marker_size"] for c in chains]
        legend_kwargs = self.parent.config["legend_kwargs"]
        legend_artists = self.parent.config["legend_artists"]
        legend_color_text = self.parent.config["legend_color_text"]
        legend_location = self.parent.config["legend_location"]

        if legend_location is None:
            if not flip or len(parameters) > 2:
                legend_location = (0, -1)
            else:
                legend_location = (-1, 0)
        outside = legend_location[0] >= legend_location[1]
        if names is not None and legend:
            ax = axes[legend_location[0], legend_location[1]]
            if "markerfirst" not in legend_kwargs:
                # If we have legend inside a used subplot, switch marker order
                legend_kwargs["markerfirst"] = outside or not legend_artists
            linewidths2 = linewidths if legend_artists else [0] * len(linewidths)
            linestyles2 = linestyles if legend_artists else ["-"] * len(linestyles)
            marker_sizes2 = marker_sizes if legend_artists else [0] * len(linestyles)

            artists = []
            done_names = []
            final_colors = []
            for i, (n, c, ls, lw, marker, size, pp, pc) in enumerate(
                zip(names, colors, linestyles2, linewidths2, marker_styles, marker_sizes2, plot_points, plot_contours)
            ):
                if n is None or n in done_names:
                    continue
                done_names.append(n)
                final_colors.append(c)
                size = np.sqrt(size)  # plot vs scatter use size differently, hence the sqrt
                if pc and not pp:
                    artists.append(plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw))
                elif not pc and pp:
                    artists.append(plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=0, marker=marker, markersize=size))
                else:
                    artists.append(plt.Line2D((0, 1), (0, 0), color=c, ls=ls, lw=lw, marker=marker, markersize=size))

            leg = ax.legend(artists, done_names, **legend_kwargs)
            if legend_color_text:
                for text, c in zip(leg.get_texts(), final_colors):
                    text.set_weight("medium")
                    text.set_color(c)
            if not outside:
                loc = legend_kwargs.get("loc") or ""
                if isinstance(loc, str) and "right" in loc.lower():
                    vp = leg._legend_box._children[-1]._children[0]
                    vp.align = "right"

        fig.canvas.draw()
        if not inverse:
            for ax in axes[-1, :]:
                offset = ax.get_xaxis().get_offset_text()
                ax.set_xlabel("{0} {1}".format(ax.get_xlabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
                offset.set_visible(False)
            for ax in axes[:, 0]:
                offset = ax.get_yaxis().get_offset_text()
                ax.set_ylabel("{0} {1}".format(ax.get_ylabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
                offset.set_visible(False)
        elif inverse:
            for ax in axes[0, :]:
                offset = ax.get_xaxis().get_offset_text()
                ax.set_xlabel("{0} {1}".format(ax.get_xlabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
                offset.set_visible(False)
            for ax in axes[:, -1]:
                offset = ax.get_yaxis().get_offset_text()
                ax.set_ylabel("{0} {1}".format(ax.get_ylabel(), "[{0}]".format(offset.get_text()) if offset.get_text() else ""))
                offset.set_visible(False)

        dpi = 300
        if watermark:
            if flip and len(parameters) == 2:
                ax = axes[-1, 0]
            else:
                ax = None
            self._add_watermark(fig, ax, figsize, watermark, dpi=dpi)

        if filename is not None:
            if isinstance(filename, str):
                filename = [filename]
            for f in filename:
                self._save_fig(fig, f, dpi)
        if display:
            plt.show()

        return fig

    def _plot_bars(self, ax, parameter, chain, flip=False, inverse=False, summary=False):  # pragma: no cover

        # Get values from config
        colour = chain.config["color"]
        linestyle = chain.config["linestyle"]
        bar_shade = chain.config["bar_shade"]
        linewidth = chain.config["linewidth"]
        bins = chain.config["bins"]
        smooth = chain.config["smooth"]
        kde = chain.config["kde"]
        zorder = chain.config["zorder"]
        title_size = self.parent.config["label_font_size"]
        chain_row = chain.get_data(parameter)
        weights = chain.weights
        if smooth or kde:
            xs, ys, _ = self.parent.analysis._get_smoothed_histogram(chain, parameter, pad=True)
            if flip:
                ax.plot(ys, xs, color=colour, ls=linestyle, lw=linewidth, zorder=zorder)
            else:
                ax.plot(xs, ys, color=colour, ls=linestyle, lw=linewidth, zorder=zorder)
        else:
            if flip:
                orientation = "horizontal"
            else:
                orientation = "vertical"
            if chain.grid:
                bins = get_grid_bins(chain_row)
            else:
                bins, smooth = get_smoothed_bins(smooth, bins, chain_row, weights)
            hist, edges = np.histogram(chain_row, bins=bins, density=True, weights=weights)
            if chain.power is not None:
                hist = hist ** chain.power
            edge_center = 0.5 * (edges[:-1] + edges[1:])
            xs, ys = edge_center, hist
            ax.hist(xs, weights=ys, bins=bins, histtype="step", color=colour, orientation=orientation, ls=linestyle, lw=linewidth, zorder=zorder)
        interp_type = "linear" if smooth else "nearest"
        interpolator = interp1d(xs, ys, kind=interp_type)
        if bar_shade:
            fit_values = self.parent.analysis.get_parameter_summary(chain, parameter)
            if fit_values is not None:
                lower = fit_values[0]
                upper = fit_values[2]
                if lower is not None and upper is not None:
                    if lower < xs.min():
                        lower = xs.min()
                    if upper > xs.max():
                        upper = xs.max()
                    x = np.linspace(lower, upper, 1000)
                    if flip:
                        ax.fill_betweenx(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2, zorder=zorder)
                    else:
                        ax.fill_between(x, np.zeros(x.shape), interpolator(x), color=colour, alpha=0.2, zorder=zorder)
                    if summary:
                        
                        t = self.parent.analysis.get_parameter_text(*fit_values)
                        loc = None if not inverse else -0.2
                        if isinstance(parameter, str):
                            ax.set_title(remove_square_brackets(r"%s$=%s$" % (parameter, t)), fontsize=25, y=loc)
                        else:
                            ax.set_title(r"$%s$" % t, fontsize=20, y=loc)
        return ys.max()
        
    def _get_figure(self, all_parameters, flip,inverse, figsize=(5, 5), external_extents=None, chains=None, blind=None, log_scales=None):  # pragma: no cover
            n = len(all_parameters)
            max_ticks = self.parent.config["max_ticks"]
            spacing = self.parent.config["spacing"]
            plot_hists = self.parent.config["plot_hists"]
            label_font_size = self.parent.config["label_font_size"]
            tick_font_size = self.parent.config["tick_font_size"]
            diagonal_tick_labels = self.parent.config["diagonal_tick_labels"]
            if blind is None:
                blind = []

            if chains is None:
                chains = self.parent.chains

            if not plot_hists:
                n -= 1

            if spacing is None:
                spacing = 1.0 if n < 6 else 0.0

            if n == 2 and plot_hists and flip:
                gridspec_kw = {"width_ratios": [3, 1], "height_ratios": [1, 3]}
            else:
                gridspec_kw = {}

            fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False, gridspec_kw=gridspec_kw)
            fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, wspace=0.05 * spacing, hspace=0.05 * spacing)

            extents = self._get_custom_extents(all_parameters, chains, external_extents)

            if plot_hists:
                params1 = all_parameters
                params2 = all_parameters
            else:
                params1 = all_parameters[1:]
                params2 = all_parameters[:-1]
            for i, p1 in enumerate(params1):
                for j, p2 in enumerate(params2):
                    ax = axes[i, j]
                    formatter_x = ScalarFormatter(useOffset=True)
                    formatter_x.set_powerlimits((-3, 4))
                    formatter_y = ScalarFormatter(useOffset=True)
                    formatter_y.set_powerlimits((-3, 4))

                    display_x_ticks = False
                    display_y_ticks = False
                    if i < j and not inverse:
                        ax.set_frame_on(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    elif j  < i and inverse:
                        ax.set_frame_on(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                    else:
                        logx = False
                        logy = False
                        if p1 == p2:
                            if log_scales.get(p1):
                                if flip and j == n - 1:
                                    ax.set_yscale("log")
                                    logy = True
                                else:
                                    ax.set_xscale("log")
                                    logx = True
                        else:
                            if log_scales.get(p1):
                                ax.set_yscale("log")
                                logy = True
                            if log_scales.get(p2):
                                ax.set_xscale("log")
                                logx = True
                        if (i != n - 1 and not inverse) or (inverse and (j!= n - 1 or (i==n-1 and j==n-1))) or (flip and j == n - 1):
                            if not inverse:
                                ax.set_xticks([])
                            else:
                                ax.set_yticks([])
                        else:
                            if p2 in blind:
                                ax.set_xticks([])
                            else:
                                if not inverse:
                                    display_x_ticks = True
                                elif inverse:
                                    display_y_ticks = True
                            if isinstance(p2, str):
                                if not inverse:
                                    ax.set_xlabel(p2, fontsize=label_font_size)
                                elif inverse:
                                    ax.yaxis.set_label_position("right")
                                    ax.yaxis.tick_right()
                                    ax.set_ylabel(p1, fontsize=label_font_size, labelpad=12.0)
                        if (j != 0 and not inverse) or (inverse and i!=0  and not (i==0 and j ==0)) or (plot_hists and i == 0 and not inverse) or (plot_hists and j == 0 and i!=0 and inverse):
                            if not inverse:
                                ax.set_yticks([])
                            else:
                                ax.set_xticks([])
                        else:
                            if p1 in blind:
                                ax.set_yticks([])
                            else:
                                if not inverse:
                                    display_y_ticks = True
                                elif inverse:
                                    display_x_ticks = True
                            if isinstance(p1, str):
                                if not inverse:
                                    ax.set_ylabel(p1, fontsize=label_font_size)
                                elif inverse:
                                    ax.xaxis.set_label_position("top")
                                    ax.xaxis.tick_top()
                                    ax.set_xlabel(p2, fontsize=label_font_size, labelpad=12.0)

                        if display_x_ticks:
                            if diagonal_tick_labels:
                                _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
                            _ = [l.set_fontsize(tick_font_size) for l in ax.get_xticklabels()]
                            if not logx:
                                ax.xaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                                ax.xaxis.set_major_formatter(formatter_x)
                            else:
                                ax.xaxis.set_major_locator(LogLocator(numticks=max_ticks))
                        else:
                            ax.set_xticks([])
                        if display_y_ticks:
                            if diagonal_tick_labels:
                                _ = [l.set_rotation(45) for l in ax.get_yticklabels()]
                            _ = [l.set_fontsize(tick_font_size) for l in ax.get_yticklabels()]
                            if not logy:
                                ax.yaxis.set_major_locator(MaxNLocator(max_ticks, prune="lower"))
                                ax.yaxis.set_major_formatter(formatter_y)
                            else:
                                ax.yaxis.set_major_locator(LogLocator(numticks=max_ticks))
                        else:
                            ax.set_yticks([])
                        if i != j or not plot_hists:
                            ax.set_ylim(extents[p1])
                        elif flip and i == 1:
                            ax.set_ylim(extents[p1])
                        ax.set_xlim(extents[p2])

            return fig, axes, params1, params2, extents