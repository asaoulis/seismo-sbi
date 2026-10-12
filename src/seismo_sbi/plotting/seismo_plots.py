"""Waveform and misfit figures.

:class:`MisfitsPlotting` draws observed against synthetic traces: raw, aligned on arrivals,
ordered by distance, as a moveout record section, and stacked against posterior-predictive draws.
"""

import matplotlib.pyplot as plt
import numpy as np

from collections import OrderedDict

from obspy.taup import tau
from obspy.geodetics import locations2degrees

from seismo_sbi.plotting.waveform_compare import moveout_record_section
from seismo_sbi.simulators.simulation_io import SYNTHETICS_PRE_EVENT_PAD_S

#: Vertical distance between neighbouring stacked traces, in units of the normalised amplitude.
STACKED_TRACE_SPACING = 0.9


def get_epicentral_distances_function(event_lat, event_long, station):
    lat, long = station
    return locations2degrees(event_lat, event_long, lat, long)

class MisfitsPlotting:

    def __init__(self, receivers, sampling_rate, covariance_matrix = None):
        self.receivers = receivers
        self.covariance_matrix = covariance_matrix

        self.sampling_rate = sampling_rate
        self.noise_levels = None
        if covariance_matrix is not None:
            # a covariance without a dense per-sample matrix gives no noise-level line
            try:
                covariances = covariance_matrix.covariance_matrix.reshape(-1, covariance_matrix.data_vector_length)[:, 0]
                self.noise_levels = np.sqrt(covariances)
            except (AttributeError, TypeError, ValueError):
                pass

    def raw_synthetic_misfits(self, data_vector, synthetics, figname = None):
        """Observed (black) against synthetic (copper) traces over the whole window, one row per
        station, with the noise level and each trace's chi-square when there is a covariance."""
        self._misfit_grid(data_vector, synthetics, figname)

    def arrival_synthetic_misfits(self, data_vector, synthetics, event_location, figname = None):
        """As :meth:`raw_synthetic_misfits`, each station cut to a window around its P arrival
        from ``event_location`` = (lat, lon, depth_km)."""
        arrivals = self._get_arrivals_dict(event_location)
        arrival_windows = {}
        for station_name, arrival_time in arrivals.items():
            start_idx = int(max(0, arrival_time*self.sampling_rate- 60*self.sampling_rate))
            end_idx = start_idx + int(50*self.sampling_rate + (2*arrival_time*self.sampling_rate))
            arrival_windows[station_name] = slice(start_idx, end_idx)
        self._misfit_grid(data_vector, synthetics, figname, arrival_windows)

    def _misfit_grid(self, data_vector, synthetics, figname, arrival_windows=None):
        """One row per station and a column per Z, E, N component.

        ``arrival_windows`` maps a station to the sample slice drawn, and without it the whole trace
        is drawn with its noise level.
        """
        time_series_length = self._trace_length(data_vector)

        if self.covariance_matrix is not None:
            elementwise_misfits = self.covariance_matrix.compute_loss(data_vector - synthetics, reduce=False)
            elementwise_misfits = np.reshape(elementwise_misfits, (-1, time_series_length))
            chi_squared = -2 * elementwise_misfits

        data_vector = np.reshape(data_vector, (-1, time_series_length))
        synthetics = np.reshape(synthetics, (-1, time_series_length))

        fig, axes = plt.subplots(len(self.receivers.receivers), 3, figsize=(5*3, 2*len(self.receivers.receivers)), squeeze=False)

        seismogram_index = 0
        component_to_index = {'Z':0, 'E':1, 'N':2}
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            station_name = receiver.station_name
            window = slice(None) if arrival_windows is None else arrival_windows[station_name]
            for component, comp_idx in component_to_index.items():
                ax = axes[rec_idx][comp_idx]
                if component not in receiver.components:
                    ax.axis('off')
                    continue

                data_seismogram = data_vector[seismogram_index][window]
                synthetic_seismogram = synthetics[seismogram_index][window]
                if arrival_windows is None and self.noise_levels is not None:
                    ax.hlines(self.noise_levels[seismogram_index], 0, time_series_length, color='red', linestyle='-',)

                ax.plot(data_seismogram, color='black',  label='data')
                ax.plot(synthetic_seismogram, color='#b87333',  label='mle')
                # an arrival window can start past the end of the trace
                if self.covariance_matrix is not None and data_seismogram.size:
                    trace_misfits = chi_squared[seismogram_index][window]
                    ax.text(0, np.max(data_seismogram), f'$\chi^2 = ${np.mean(trace_misfits):.2f}')

                ax.axis('off')
                ax.set_title(f'{station_name} {component}')

                seismogram_index += 1

        self._finish(fig, figname)

    def plot_ordered_stacked_traces(self, data_vector, synthetics, event_location, figname=None,
                                    components=('Z', 'E', 'N'), time_window_s=None):
        """Observed (black) against synthetic (copper) traces, one column per entry of ``components``
        (Z, E, N by default), stations ordered by P arrival from the top.

        Every trace is scaled by the largest observed amplitude drawn. ``time_window_s`` = (start,
        end) in seconds limits the window drawn.
        """
        trace_index_map = self._trace_index_map()
        ordered_by_comp = self._pairs_by_arrival(event_location, trace_index_map, components)
        time_series_length = self._trace_length(data_vector)

        data_matrix = np.reshape(data_vector, (-1, time_series_length))
        synthetics_matrix = np.reshape(synthetics, (-1, time_series_length))

        drawn = [trace_index_map[pair] for comp in components for pair in ordered_by_comp[comp]]
        max_abs = np.max(np.abs(data_matrix[drawn])) if drawn else 1.0
        if max_abs == 0:
            max_abs = 1.0

        t = np.arange(time_series_length) / float(self.sampling_rate)
        fig, axes = self._stacked_axes(len(components), 5.04 * len(components))

        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            if len(pairs) == 0:
                ax.axis('off')
                continue
            offsets = np.arange(len(pairs)) * STACKED_TRACE_SPACING
            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                d = data_matrix[flat_idx] / max_abs
                s = synthetics_matrix[flat_idx] / max_abs
                y0 = offsets[k]
                ax.plot(t, y0 + d, color='black')
                ax.plot(t, y0 + s, color='#b87333')
            self._style_stacked_axis(ax, offsets, pairs, comp, 'Time [s]')
            if time_window_s is not None:
                ax.set_xlim(time_window_s)

        # y tick labels on the first column only. X tick labels on the others
        for i, ax in enumerate(axes):
            if i > 0:
                ax.tick_params(labelleft=False)
            elif len(axes) > 1:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel('')

        plt.tight_layout()
        self._finish(fig, figname, dpi=300, transparent=True)

    def _get_arrivals_dict(self, event_location):
        event_lat, event_lon, depth = event_location
        taup = tau.TauPyModel(model='prem')
        arrivals = {}

        for station_details in self.receivers.iterate():

            station = station_details.latitude, station_details.longitude
            distance = get_epicentral_distances_function(event_lat, event_lon, station)
            station_arrival = taup.get_travel_times(source_depth_in_km=depth,
                                                    distance_in_degree=distance
                                                    )      
            arrivals[station_details.station_name] = station_arrival[0].time + SYNTHETICS_PRE_EVENT_PAD_S
        
        return arrivals

    # ---------------------------------------------------------------- shared steps
    def _trace_length(self, vector):
        """Samples per trace of a flat data vector over every receiver component."""
        n_traces = len([component for receiver in self.receivers.iterate() for component in receiver.components])
        return int(len(vector) // n_traces)

    def _trace_index_map(self):
        """``{(station, component): index}`` of each trace in the flat data vector, receiver by
        receiver in each receiver's component order."""
        trace_index_map = {}
        idx = 0
        for receiver in self.receivers.iterate():
            for comp in receiver.components:
                trace_index_map[(receiver.station_name, comp)] = idx
                idx += 1
        return trace_index_map

    def _pairs_by_arrival(self, event_location, trace_index_map, components):
        """``{component: [(station, component), ...]}`` with the stations in order of P arrival."""
        arrivals = self._get_arrivals_dict(event_location)
        ordered_stations = sorted(arrivals.keys(), key=lambda k: arrivals[k])
        return {
            comp: [(st, comp) for st in ordered_stations if (st, comp) in trace_index_map]
            for comp in components
        }

    @staticmethod
    def _ensemble_colors(ensembles, colors):
        """``{ensemble name: colour}`` from a list (cycled in ensemble order) or a dict of colours."""
        if colors is None:
            palette = ['blue', 'red', 'plum']
            return {name: palette[i % len(palette)] for i, name in enumerate(ensembles.keys())}
        if isinstance(colors, dict):
            return colors
        return {name: colors[i % len(colors)] for i, name in enumerate(ensembles.keys())}

    @staticmethod
    def _stacked_axes(n_columns, width):
        fig, axes = plt.subplots(1, n_columns, figsize=(width, 0.7 * 8.0))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        return fig, axes

    @staticmethod
    def _style_stacked_axis(ax, offsets, pairs, component, xlabel):
        """Station names as y ticks, the earliest arrival at the top, no frame."""
        ax.set_yticks(offsets)
        ax.set_yticklabels([st for (st, _) in pairs])
        ax.set_xlabel(xlabel)
        amp_pad = 1.1
        ax.set_ylim([-amp_pad, offsets[-1] + amp_pad])
        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        ax.set_title(f'{component}')

    @staticmethod
    def _finish(fig, figname, **savefig_kwargs):
        """Save to ``figname`` when given, otherwise show. Then close."""
        if figname is not None:
            fig.savefig(figname, **savefig_kwargs)
            fig.clear()
        else:
            plt.show()
        plt.close()

    # ---------------------------------------------------------------- record section
    def _reshape_to_cube(self, flat):
        """``(n_traces * T,)`` flat vector -> ``(N_stations, C, T)`` cube.

        Uses the receivers' own (receiver-major, then component) order: the same layout
        the simulator flattens into, so this inverts it exactly. Requires a uniform
        component set across the (restricted) receivers, which is what the dataloader's
        ``load_event_subset(..., stacked=True)`` produces.
        """
        flat = np.asarray(flat, float)
        comps = [list(r.components) for r in self.receivers.iterate()]
        n_sta = len(comps)
        if n_sta == 0:
            raise ValueError("no receivers to reshape against")
        if len({tuple(c) for c in comps}) != 1:
            raise ValueError(f"record section needs a uniform component set, got {set(map(tuple, comps))}")
        n_comp = len(comps[0])
        time_length = int(flat.shape[-1] // (n_sta * n_comp))
        return flat.reshape(-1, n_sta, n_comp, time_length), comps[0]

    def plot_record_section(self, observation, event_location, *, deterministic=None,
                            ensembles=None, channel_mask=None, max_samples=40,
                            use_arrivals=False, title=None, figname=None, **kwargs):
        """Moveout record section of the observation vs any number of overlays.

        Adapter between this class's flat-vector / ``Receivers`` world and the pure-array
        :func:`seismo_sbi.plotting.waveform_compare.moveout_record_section`.

        Parameters
        ----------
        observation : flat 1-D vector
        deterministic : dict, optional
            ``label -> flat vector``: single synthetics (best-fit MT, reference MTs).
        ensembles : dict, optional
            ``label -> (n_samples, data_len)``: posterior predictive clouds. Subsampled to
            ``max_samples`` before rendering.
        channel_mask : ``(N, C)`` bool, optional
            ``False`` = QA-dropped channel (greyed and excluded from the annotations).
        use_arrivals : bool
            Compute taup P-arrivals so ``order_by='arrival'`` / ``window=('arrival', ...)``
            can be used. Costs a taup call per station.
        **kwargs
            Passed straight to ``moveout_record_section`` (``order_by``, ``y_scale``,
            ``normalise``, ``layout``, ``window``, ``ensemble_style``, ...).
        """
        obs_cube, components = self._reshape_to_cube(observation)
        obs_cube = obs_cube[0]
        overlays = OrderedDict()
        for label, vec in (deterministic or {}).items():
            overlays[label] = self._reshape_to_cube(vec)[0][0]
        rng = np.random.default_rng(kwargs.get("seed", 0))
        for label, arr in (ensembles or {}).items():
            arr = np.asarray(arr, float)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            if arr.shape[0] > max_samples:
                # keep member 0 FIRST: `select_best_synthetics` returns the ensemble
                # ordered best-first, and the 'band+best' style draws member 0.
                rest = rng.choice(np.arange(1, arr.shape[0]), max_samples - 1, replace=False)
                arr = arr[np.concatenate(([0], rest))]
            overlays[label] = self._reshape_to_cube(arr)[0]

        station_names = [r.station_name for r in self.receivers.iterate()]
        coords = np.array([[r.latitude, r.longitude] for r in self.receivers.iterate()])
        arrivals = self._get_arrivals_dict(event_location) if use_arrivals else None

        return moveout_record_section(
            obs_cube, overlays, station_names, coords, event_location,
            components=components, sampling_rate=self.sampling_rate,
            arrivals=arrivals, channel_mask=channel_mask, title=title,
            figname=figname, **{k: v for k, v in kwargs.items() if k != "seed"})


    def plot_posterior_predictive_stacked_traces(
        self,
        observation,
        ensembles,
        event_location,
        colors=None,
        max_samples=100,
        alpha=0.08,
        seed=None,
        figname=None,
        per_trace=False,
    ):
        """
        Posterior predictive check plot akin to plot_ordered_stacked_traces:
        - Columns per component (Z, E, N), stations ordered by earliest P-arrival.
        - Plots the observation (black) and random samples from multiple ensembles.
        - Adds provided metrics per ensemble as a small textbox.

        per_trace : bool
            If False (default) every trace is normalised by the single global observation
            peak (amplitudes are comparable across stations). If True each trace is scaled by
            its own observation peak, so the waveform shape fit is visible on quiet/distant
            stations that a global scale flattens.

        Parameters
        ----------
        observation : np.ndarray
            1D vector of length data_vector_length.
        ensembles : Dict[str, np.ndarray]
            Mapping ensemble_name -> samples array with shape [N_samples, data_vector_length].

        event_location : tuple
            (lat, lon, depth_km).
        colors : List[str] or Dict[str, str], optional
            Colors per ensemble. If list, assigned in order of ensembles' keys.
            Default: ['blue', 'red', 'plum'].
        max_samples : int
            Max number of samples drawn per ensemble for plotting.
        alpha : float
            Line alpha for samples.
        seed : int or None
            RNG seed for reproducible subsampling.
        figname : str or None
            If provided, saves figure to this path. Otherwise shows interactively.
        """
        rng = np.random.default_rng(seed)
        components = ['Z', 'E', 'N']
        trace_index_map = self._trace_index_map()
        ordered_by_comp = self._pairs_by_arrival(event_location, trace_index_map, components)
        time_series_length = self._trace_length(observation)

        obs_matrix = np.reshape(observation, (-1, time_series_length))
        ens_matrices = {
            name: np.reshape(arr, (arr.shape[0], -1, time_series_length))
            for name, arr in ensembles.items()
        }

        global_max = np.max(np.abs(obs_matrix)) if obs_matrix.size else 1.0
        if global_max == 0:
            global_max = 1.0

        def _norm(flat_idx):
            if not per_trace:
                return global_max
            m = float(np.max(np.abs(obs_matrix[flat_idx])))
            return m if m > 0 else 1.0

        t = np.arange(time_series_length) / float(self.sampling_rate)
        color_map = self._ensemble_colors(ensembles, colors)
        fig, axes = self._stacked_axes(3, 12.6 * 1.2)

        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            if len(pairs) == 0:
                ax.axis('off')
                continue

            offsets = np.arange(len(pairs)) * STACKED_TRACE_SPACING

            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                d = obs_matrix[flat_idx] / _norm(flat_idx)
                y0 = offsets[k]
                ax.plot(t, y0 + d, color='black', linewidth=1.2, zorder=3, label='Observation')

            for name, sample_cube in ens_matrices.items():
                color = color_map.get(name, 'gray')
                N = sample_cube.shape[0]
                if N == 0:
                    continue
                take = N if N <= max_samples else max_samples
                sel = rng.choice(N, size=take, replace=False) if N > take else np.arange(N)
                for k, (st, c) in enumerate(pairs):
                    flat_idx = trace_index_map[(st, c)]
                    y0 = offsets[k]
                    samples = sample_cube[sel, flat_idx, :] / _norm(flat_idx)
                    for row in samples:
                        ax.plot(t, y0 + row, color=color, alpha=alpha, linewidth=0.6, zorder=2)

            self._style_stacked_axis(ax, offsets, pairs, comp, 'Time [s]')

        for i, ax in enumerate(axes):
            if i > 0:
                ax.tick_params(labelleft=False)
            ax.set_ylabel('')
        if len(ensembles) > 0:
            legend_elements = [plt.Line2D([0], [0], color='black', lw=2, label='Observation')]
            legend_elements += [
                plt.Line2D([0], [0], color=color_map.get(name, 'gray'), lw=2, label=name)
                for name in ensembles.keys()
            ]
            axes[-1].legend(
                handles=legend_elements,
                loc='upper right',
                fontsize='small',
                framealpha=0.8,
            )
        plt.tight_layout()
        self._finish(fig, figname, dpi=300, transparent=True)

    def plot_posterior_predictive_stacked_traces_zoomin(
        self,
        observation,
        ensembles,
        event_location,
        M=50,
        colors=None,
        max_samples=100,
        alpha=0.08,
        seed=None,
        figname=None,
    ):
        """Posterior predictive stacked traces zoomed around per-trace peak.

        Similar to plot_posterior_predictive_stacked_traces, but for each
        station-component trace we select a window of length ``M`` samples
        centered on the maximum absolute amplitude of the observation for
        that trace. The x-axis is expressed as relative time to the peak
        amplitude (seconds). Centering is done per-station, per-component.
        """
        rng = np.random.default_rng(seed)
        components = ['Z', 'E', 'N']
        trace_index_map = self._trace_index_map()
        ordered_by_comp = self._pairs_by_arrival(event_location, trace_index_map, components)
        time_series_length = self._trace_length(observation)

        obs_matrix = np.reshape(observation, (-1, time_series_length))
        ens_matrices = {
            name: np.reshape(arr, (arr.shape[0], -1, time_series_length))
            for name, arr in ensembles.items()
        }

        max_abs = np.max(np.abs(obs_matrix)) if obs_matrix.size else 1.0
        if max_abs == 0:
            max_abs = 1.0

        color_map = self._ensemble_colors(ensembles, colors)
        half_M = M // 2
        fig, axes = self._stacked_axes(3, 12.6 * 1.2)

        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            if len(pairs) == 0:
                ax.axis('off')
                continue

            offsets = np.arange(len(pairs)) * STACKED_TRACE_SPACING

            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                obs_trace = obs_matrix[flat_idx]
                peak_idx = int(np.argmax(np.abs(obs_trace)))
                start = max(0, peak_idx - half_M)
                end = min(time_series_length, start + M)
                # keep the window M samples long near the end of the trace
                if end - start < M:
                    start = max(0, end - M)
                win_obs = obs_trace[start:end] / max_abs

                local_len = end - start
                t_local = (np.arange(local_len) - (peak_idx - start)) / float(self.sampling_rate)

                y0 = offsets[k]
                ax.plot(t_local, y0 + win_obs, color='black', linewidth=1.2, zorder=3,
                        label='Observation' if (i == 0 and k == 0) else None)

                for name, sample_cube in ens_matrices.items():
                    color = color_map.get(name, 'gray')
                    N = sample_cube.shape[0]
                    if N == 0:
                        continue
                    take = N if N <= max_samples else max_samples
                    sel = rng.choice(N, size=take, replace=False) if N > take else np.arange(N)
                    samples = sample_cube[sel, flat_idx, start:end] / max_abs
                    for row in samples:
                        ax.plot(t_local, y0 + row, color=color, alpha=alpha, linewidth=0.6, zorder=2,
                                label=name if (i == 0 and k == 0) else None)

            self._style_stacked_axis(ax, offsets, pairs, comp, 'Time relative to peak [s]')

        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles += h
            labels += l
        if handles:
            seen = set()
            uniq = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
            axes[-1].legend([h for h, _ in uniq], [l for _, l in uniq], loc='upper right', fontsize='small', framealpha=0.85)

        for i, ax in enumerate(axes):
            if i > 0:
                ax.tick_params(labelleft=False)
            ax.set_ylabel('')

        plt.tight_layout()
        self._finish(fig, figname, dpi=300, transparent=True)
