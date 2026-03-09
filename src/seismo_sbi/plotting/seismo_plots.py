import matplotlib.pyplot as plt   
import numpy as np
import math as m

from obspy.taup import tau
from obspy.geodetics import locations2degrees

def get_epicentral_distances_function(event_lat, event_long, station):
    lat, long = station
    return locations2degrees(event_lat, event_long, lat, long)

def plot_stacked_waveforms(receivers, flattened_seismogram_array, figname = None):
    num_receivers = len(receivers)
    all_components =  [component for receiver in receivers for component in receiver.components]
    time_series_length = int(flattened_seismogram_array.shape[0]//len(all_components))
    t_axis = list(range(time_series_length))
    receiver_seismograms = np.reshape(flattened_seismogram_array, (-1, time_series_length))

    fig = plt.figure(figsize=(15, num_receivers//2))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(
        plt.cycler("color", plt.cm.prism(np.linspace(0, 0.4, num_receivers)))
    )

    seismogram_scale = np.mean(np.abs(receiver_seismograms))*10
    offsets = np.linspace(0,seismogram_scale*num_receivers, num_receivers)
    seismogram_index = 0
    for offset, receiver in zip(offsets, receivers):
        station_name = receiver.station_name
        start_time = 0
        for component in receiver.components:
            seismogram = receiver_seismograms[seismogram_index]
            ax.plot(np.array(t_axis) + start_time, seismogram + offset)
            start_time += time_series_length + time_series_length//10
            seismogram_index += 1

        ax.text(
            t_axis[0] + 2,
            offset + 0.35 * seismogram_scale,
            f"{station_name}",
            fontsize=10,
            weight="bold",
            color="red",
        )

    if figname is not None:
        fig.savefig(figname)
        fig.clear()
    else:
        plt.show()
    plt.close()

class MisfitsPlotting:

    def __init__(self, receivers, sampling_rate, covariance_matrix = None):
        self.receivers = receivers
        self.covariance_matrix = covariance_matrix

        self.sampling_rate = sampling_rate
        self.noise_levels = None
        if covariance_matrix is not None:

            try:
                covariances = covariance_matrix.covariance_matrix.reshape(-1, covariance_matrix.data_vector_length)[:, 0]
                self.noise_levels = np.sqrt(covariances)
            except:
                pass

    def raw_synthetic_misfits(self, data_vector, synthetics, figname = None):


        all_components =  [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(data_vector.shape[0]//len(all_components))

        if self.covariance_matrix is not None:
            elementwise_misfits = self.covariance_matrix.compute_loss(data_vector - synthetics, reduce=False)
            elementwise_misfits = np.reshape(elementwise_misfits, (-1, time_series_length))
            chi_squared = -2 * elementwise_misfits

        data_vector = np.reshape(data_vector, (-1, time_series_length))
        synthetics = np.reshape(synthetics, (-1, time_series_length))


        fig, axes = plt.subplots(len(self.receivers.receivers), 3, figsize=(5*3, 2*len(self.receivers.receivers)))

        seismogram_index = 0
        component_to_index = {'Z':0, 'E':1, 'N':2}
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            station_name = receiver.station_name
            for component, comp_idx in component_to_index.items():
                ax = axes[rec_idx][comp_idx]
                if component not in receiver.components:
                    ax.axis('off')
                    continue
                
                data_seismogram = data_vector[seismogram_index]
                synthetic_seismogram = synthetics[seismogram_index]
                if self.covariance_matrix is not None:
                    trace_misfits = chi_squared[seismogram_index]
                    ax.text(0, np.max(data_seismogram), f'$\chi^2 = ${np.mean(trace_misfits):.2f}')
                if self.noise_levels is not None:
                    ax.hlines(self.noise_levels[seismogram_index], 0, time_series_length, color='red', linestyle='-',)
                    
                ax.plot(data_seismogram, color='black',  label='data')
                ax.plot(synthetic_seismogram, color='#b87333',  label='mle')

                ax.axis('off')
                ax.set_title(f'{station_name} {component}')

                seismogram_index += 1

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close()

    def arrival_synthetic_misfits(self, data_vector, synthetics, event_location, figname = None):

        arrivals = self._get_arrivals_dict(event_location)

        all_components =  [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(data_vector.shape[0]//len(all_components))

        if self.covariance_matrix is not None:
            elementwise_misfits = self.covariance_matrix.compute_loss(data_vector - synthetics, reduce=False)
            elementwise_misfits = np.reshape(elementwise_misfits, (-1, time_series_length))
            chi_squared = -2 * elementwise_misfits


        data_vector = np.reshape(data_vector, (-1, time_series_length))
        synthetics = np.reshape(synthetics, (-1, time_series_length))

        fig, axes = plt.subplots(len(self.receivers.receivers), 3, figsize=(5*3, 2*len(self.receivers.receivers)))

        seismogram_index = 0
        component_to_index = {'Z':0, 'E':1, 'N':2}
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            station_name = receiver.station_name
            arrival_time = arrivals[station_name]
            start_idx = int(max(0, arrival_time*self.sampling_rate- 60*self.sampling_rate))
            end_idx = start_idx + int(50*self.sampling_rate + (2*arrival_time*self.sampling_rate))
            arrival_slice = slice(start_idx, end_idx)
            for component, comp_idx in component_to_index.items():
                ax = axes[rec_idx][comp_idx]
                if component not in receiver.components:
                    ax.axis('off')
                    continue
                data_seismogram = data_vector[seismogram_index][arrival_slice]
                synthetic_seismogram = synthetics[seismogram_index][arrival_slice]

                ax = axes[rec_idx][component_to_index[component]]
                ax.plot(data_seismogram, color='black',  label='data')
                ax.plot(synthetic_seismogram, color='#b87333',  label='mle')
                if self.covariance_matrix is not None:
                    trace_misfits = chi_squared[seismogram_index][arrival_slice]
                    try:
                        ax.text(0, np.max(data_seismogram), f'$\chi^2 = ${np.mean(trace_misfits):.2f}')
                    except:
                        pass

                ax.axis('off')
                ax.set_title(f'{station_name} {component}')

                seismogram_index += 1

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close()
    
    def moveout_synthetic_misfits(self, data_vector, synthetics, event_location, figname = None):
       
        arrivals = self._get_arrivals_dict(event_location)
        order_of_stations = sorted(arrivals, key=arrivals.get, reverse=True)

        all_components =  [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(data_vector.shape[0]//len(all_components))

        if self.covariance_matrix is not None:
            elementwise_misfits = self.covariance_matrix.compute_loss(data_vector - synthetics, reduce=False)
            elementwise_misfits = np.reshape(elementwise_misfits, (-1, time_series_length))
            chi_squared = -2 * elementwise_misfits


        data_vector = np.reshape(data_vector, (-1, time_series_length))
        synthetics = np.reshape(synthetics, (-1, time_series_length))

        fig, axes = plt.subplots(1, 3, figsize=(5*3, len(self.receivers.receivers)))

        seismogram_index = 0
        component_to_index = {'Z':0, 'E':1, 'N':2}
        locations = np.linspace(0, len(order_of_stations), len(order_of_stations))
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            station_name = receiver.station_name
            arrival_time = arrivals[station_name]
            start_idx = int(max(0, arrival_time*self.sampling_rate- 60*self.sampling_rate))
            end_idx = start_idx + int(50*self.sampling_rate + (2*arrival_time*self.sampling_rate))
            arrival_slice = slice(start_idx, end_idx)
            for component, comp_idx in component_to_index.items():
                ax = axes[comp_idx]
                if component not in receiver.components:

                    continue
                data_seismogram = data_vector[seismogram_index][arrival_slice]
                synthetic_seismogram = synthetics[seismogram_index][arrival_slice]

                ax = axes[component_to_index[component]]
                data_max = 2.5*np.max(data_seismogram)
                y_pos = locations[order_of_stations.index(station_name)]
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + data_seismogram/data_max, color='black',  label='data')
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + synthetic_seismogram/data_max, color='#b87333',  label='mle')

                seismogram_index += 1
        for component ,ax in zip(list(component_to_index.keys()), axes):
            ax.set_yticks(locations)
            ax.set_yticklabels(order_of_stations)
            ax.set_ylabel('Station')
            ax.set_xlabel('Time [s]')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f'{component} moveout')
            

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close() 

    def moveout_synthetic_misfits_wrapped(self, data_vector, synthetics, event_location, selected_receivers, width_ratios=[1, 2.5], figname = None):
       
        arrivals = self._get_arrivals_dict(event_location)
        order_of_stations = sorted(arrivals, key=arrivals.get, reverse=True)
        station_components = {receiver.station_name: receiver.components for receiver in self.receivers.iterate()}
        order_of_stations = sum([[f'{station} {component}' for component in station_components[station]] for station in order_of_stations if station in selected_receivers], [])
        all_components =  [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(data_vector.shape[0]//len(all_components))

        if self.covariance_matrix is not None:
            elementwise_misfits = self.covariance_matrix.compute_loss(data_vector - synthetics, reduce=False)
            elementwise_misfits = np.reshape(elementwise_misfits, (-1, time_series_length))
            chi_squared = -2 * elementwise_misfits


        data_vector = np.reshape(data_vector, (-1, time_series_length))
        synthetics = np.reshape(synthetics, (-1, time_series_length))

        fig, axes = plt.subplots(1, 2, figsize=(5*2, len(order_of_stations)//2), gridspec_kw={'width_ratios': width_ratios})
        seismogram_index = 0
        component_to_index = {'Z':0, 'E':1, 'N':2}
        locations = np.linspace(0, len(order_of_stations), len(order_of_stations)//2)
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            station_name = receiver.station_name
            arrival_time = arrivals[station_name]
            start_idx = int(max(0, 1.5*arrival_time*self.sampling_rate- 60*self.sampling_rate))
            end_idx = start_idx + int(110*self.sampling_rate + (1.1*arrival_time*self.sampling_rate))
            arrival_slice = slice(start_idx, end_idx)
            for component, comp_idx in component_to_index.items():
                station_comp_name = f"{station_name} {component}"
                
                if component not in receiver.components:
                    continue
                if station_comp_name not in order_of_stations:
                    seismogram_index += 1
                    continue

                index = order_of_stations.index(station_comp_name)
                ax_idx = 1 - (index // (len(order_of_stations)//2))
                ax = axes[ax_idx]
                data_seismogram = data_vector[seismogram_index][arrival_slice]
                synthetic_seismogram = synthetics[seismogram_index][arrival_slice]

                data_max = np.max(data_seismogram) *1.1
                y_pos = locations[index% len(locations)]
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + data_seismogram/data_max, color='black', label='data')
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + synthetic_seismogram/data_max, color='#b87333', label='mle')
                seismogram_index += 1
        for idx ,ax in enumerate(axes):
            ax.set_yticks(locations)
            ax.set_yticklabels(np.array(order_of_stations).reshape(2, -1)[1-idx, :])
            ax.set_ylim([-1.5, len(order_of_stations) + 1.5])
            if idx == 0:
                ax.set_ylabel('Station component')
            ax.set_xlabel('Time after origin [s]')
            for spine in ax.spines.values():
                spine.set_visible(False)
            
        plt.tight_layout()
        if figname is not None:
            fig.savefig(figname, dpi=200, transparent=True)
            fig.clear()
        else:
            plt.show()
        plt.close() 

    def moveout_synthetic_samples(self, data_vector, synthetics, event_location, selected_receivers, sampled_synthetics, figname = None):
       
        arrivals = self._get_arrivals_dict(event_location)
        order_of_stations = sorted(arrivals, key=arrivals.get, reverse=True)
        order_of_stations = [station for station in order_of_stations if station in selected_receivers]

        all_components =  [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(data_vector.shape[0]//len(all_components))

        if self.covariance_matrix is not None:
            elementwise_misfits = self.covariance_matrix.compute_loss(data_vector - synthetics, reduce=False)
            elementwise_misfits = np.reshape(elementwise_misfits, (-1, time_series_length))
            chi_squared = -2 * elementwise_misfits


        data_vector = np.reshape(data_vector, (-1, time_series_length))
        synthetics = np.reshape(synthetics, (-1, time_series_length))
        samples = np.reshape(sampled_synthetics, (sampled_synthetics.shape[0], -1, time_series_length))

        fig, axes = plt.subplots(1, 3, figsize=(5*3, len(self.receivers.receivers)))

        seismogram_index = 0
        component_to_index = {'Z':0, 'E':1, 'N':2}
        locations = np.linspace(0, len(order_of_stations), len(order_of_stations))
        for rec_idx, receiver in enumerate(self.receivers.iterate()):
            station_name = receiver.station_name
            arrival_time = arrivals[station_name]
            start_idx = int(max(0, arrival_time*self.sampling_rate- 60*self.sampling_rate))
            end_idx = start_idx + int(50*self.sampling_rate + (2*arrival_time*self.sampling_rate))
            arrival_slice = slice(start_idx, end_idx)
            for component, comp_idx in component_to_index.items():
                ax = axes[comp_idx]
                if component not in receiver.components:
                    continue
                if station_name not in order_of_stations:
                    seismogram_index += 1
                    continue

                data_seismogram = data_vector[seismogram_index][arrival_slice]
                synthetic_seismogram = synthetics[seismogram_index][arrival_slice]
                selected_samples = samples[:, seismogram_index, arrival_slice]

                ax = axes[component_to_index[component]]
                data_max = 2.5*np.max(data_seismogram)
                y_pos = locations[order_of_stations.index(station_name)]
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + data_seismogram/data_max, color='black',  label='data')
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + synthetic_seismogram/data_max, color='#b87333',  label='mle')
                ax.plot(60 + np.arange(start_idx, end_idx), y_pos + selected_samples.T/data_max, color='gray',  alpha = 0.1)
                seismogram_index += 1
        for component ,ax in zip(list(component_to_index.keys()), axes):
            ax.set_yticks(locations)
            ax.set_yticklabels(order_of_stations)
            ax.set_ylabel('Station')
            ax.set_xlabel('Time [s]')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f'{component} moveout')
            

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close() 
    
    def plot_ordered_stacked_traces(self, data_vector, synthetics, event_location, figname=None):
        """
        Plot all component traces, split across three columns (Z, E, N).
        - Stations ordered by earliest P-arrival (top to bottom).
        - One axis per component; traces vertically offset within each axis.
        - Global amplitude normalization based on data.
        - Slightly smaller figure (~30% smaller) to match other plots; minimal row spacing.
        - Keep y tick labels only on first column; keep x tick labels only on 2nd and 3rd columns.
        """
        # Order stations by arrival time
        arrivals = self._get_arrivals_dict(event_location)
        ordered_stations = sorted(arrivals.keys(), key=lambda k: arrivals[k])

        # Build mapping from (station, component) -> flattened index
        all_components_list = [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(data_vector.shape[0] // len(all_components_list))
        trace_index_map = {}
        idx = 0
        for receiver in self.receivers.iterate():
            for comp in ['Z', 'E', 'N']:
                if comp in receiver.components:
                    trace_index_map[(receiver.station_name, comp)] = idx
                    idx += 1

        # Reshape vectors
        data_matrix = np.reshape(data_vector, (-1, time_series_length))
        synthetics_matrix = np.reshape(synthetics, (-1, time_series_length))

        # Global normalization from data
        max_abs = np.max(np.abs(data_matrix)) if data_matrix.size else 1.0
        if max_abs == 0:
            max_abs = 1.0

        # Prepare per-component ordered lists
        components = ['Z', 'E', 'N']
        ordered_by_comp = {
            comp: [(st, comp) for st in ordered_stations if (st, comp) in trace_index_map]
            for comp in components
        }
        n_per_comp = {comp: len(ordered_by_comp[comp]) for comp in components}
        n_max = max(n_per_comp.values()) if n_per_comp else 0

        # Time axis
        t = np.arange(time_series_length) / float(self.sampling_rate)

        # Figure sizing: reduce by ~30% and reduce vertical spacing between stacked traces
        spacing = 0.9  # tighter stacking between rows
        height_base = 8.0
        height = 0.7 * height_base
        width = 12.6* 1.2  # 30% smaller than 18.0
        fig, axes = plt.subplots(1, 3, figsize=(width, height))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Plot per component
        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            n_traces = len(pairs)
            if n_traces == 0:
                ax.axis('off')
                continue
            offsets = np.arange(n_traces) * spacing
            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                d = data_matrix[flat_idx] / max_abs
                s = synthetics_matrix[flat_idx] / max_abs
                y0 = offsets[k]
                ax.plot(t, y0 + d, color='black')
                ax.plot(t, y0 + s, color='#b87333')
            ax.set_yticks(offsets)
            ax.set_yticklabels([st for (st, _) in pairs])
            ax.set_xlabel('Time [s]')
            # Add padding so largest trace does not clip; invert y so earliest arrivals at top
            amp_pad = 1.1
            ax.set_ylim([-amp_pad, (offsets[-1] + amp_pad) if n_traces > 0 else amp_pad])
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_title(f'{comp}')

        # Keep y tick labels only on first column; keep x tick labels on 2nd and 3rd only
        for i, ax in enumerate(axes):
            if i == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(labelleft=False)

        # Remove y-axis label entirely
        for ax in axes:
            ax.set_ylabel('')

        plt.tight_layout()
        if figname is not None:
            fig.savefig(figname, dpi=300, transparent=True)
            fig.clear()
        else:
            plt.show()
        plt.close()
    
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
            # TODO: Replace hardcoded 60 with the arbitrary offset we use to ensure filtered
            # synthetics and data are not cut out of the window
            arrivals[station_details.station_name] = station_arrival[0].time + 60
        
        return arrivals

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
    ):
        """
        Posterior predictive check plot akin to plot_ordered_stacked_traces:
        - Columns per component (Z, E, N), stations ordered by earliest P-arrival.
        - Plots the observation (black) and random samples from multiple ensembles.
        - Adds provided metrics per ensemble as a small textbox.

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
            If provided, saves figure to this path; otherwise shows interactively.
        """
        rng = np.random.default_rng(seed)
        arrivals = self._get_arrivals_dict(event_location)
        ordered_stations = sorted(arrivals.keys(), key=lambda k: arrivals[k])

        # Build flattened-trace index map following the data layout
        # (receiver-major, then receiver.components order).
        all_components_list = [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(len(observation) // len(all_components_list))

        trace_index_list = []
        idx = 0
        for receiver in self.receivers.iterate():
            for comp in receiver.components:
                trace_index_list.append((receiver.station_name, comp, idx))
                idx += 1
        trace_index_map = {(st, comp): i for (st, comp, i) in trace_index_list}

        # Reshape observation and ensembles
        obs_matrix = np.reshape(observation, (-1, time_series_length))
        ens_matrices = {
            name: np.reshape(arr, (arr.shape[0], -1, time_series_length))
            for name, arr in ensembles.items()
        }

        # Global normalization based on observation
        max_abs = np.max(np.abs(obs_matrix)) if obs_matrix.size else 1.0
        if max_abs == 0:
            max_abs = 1.0

        # Component-specific ordered lists
        components = ['Z', 'E', 'N']
        ordered_by_comp = {
            comp: [(st, comp) for st in ordered_stations if (st, comp) in trace_index_map]
            for comp in components
        }

        # Time axis
        t = np.arange(time_series_length) / float(self.sampling_rate)

        # Colors per ensemble
        if colors is None:
            palette = ['blue', 'red', 'plum']
            ens_names = list(ensembles.keys())
            color_map = {name: palette[i % len(palette)] for i, name in enumerate(ens_names)}
        elif isinstance(colors, dict):
            color_map = colors
        else:
            ens_names = list(ensembles.keys())
            color_map = {name: colors[i % len(colors)] for i, name in enumerate(ens_names)}

        # Figure
        spacing = 0.9
        height = 0.7 * 8.0
        width = 12.6 * 1.2
        fig, axes = plt.subplots(1, 3, figsize=(width, height))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        # Plot per component
        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            n_traces = len(pairs)
            if n_traces == 0:
                ax.axis('off')
                continue

            offsets = np.arange(n_traces) * spacing

            # Draw observation (black)
            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                d = obs_matrix[flat_idx] / max_abs
                y0 = offsets[k]
                ax.plot(t, y0 + d, color='black', linewidth=1.2, zorder=3, label='Observation')

            # Draw ensembles (thin colored)
            for name, sample_cube in ens_matrices.items():
                color = color_map.get(name, 'gray')
                N = sample_cube.shape[0]
                if N == 0:
                    continue
                take = N if N <= max_samples else max_samples
                sel = rng.choice(N, size=take, replace=False) if N > take else np.arange(N)
                # For each trace, overlay selected samples
                for k, (st, c) in enumerate(pairs):
                    flat_idx = trace_index_map[(st, c)]
                    y0 = offsets[k]
                    # samples: [take, time]
                    samples = sample_cube[sel, flat_idx, :] / max_abs
                    # plot lines (vectorized loop)
                    for row in samples:
                        ax.plot(t, y0 + row, color=color, alpha=alpha, linewidth=0.6, zorder=2)

            ax.set_yticks(offsets)
            ax.set_yticklabels([st for (st, _) in pairs])
            ax.set_xlabel('Time [s]')
            amp_pad = 1.1
            ax.set_ylim([-amp_pad, (offsets[-1] + amp_pad) if n_traces > 0 else amp_pad])
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_title(f'{comp}')


        # Ticks: keep y labels only on first, x labels on 2nd and 3rd
        for i, ax in enumerate(axes):
            if i == 0:
                # ax.tick_params(labelbottom=False)
                pass
            else:
                ax.tick_params(labelleft=False)

        for ax in axes:
            ax.set_ylabel('')
        # add a legend box with ensemble names and colors
        if len(ensembles) > 0:
            legend_elements = []
            legend_elements.insert(0, plt.Line2D([0], [0], color='black', lw=2, label='Observation'))

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
        if figname is not None:
            fig.savefig(figname, dpi=300, transparent=True)
            fig.clear()
        else:
            plt.show()
        plt.close()

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
        arrivals = self._get_arrivals_dict(event_location)
        ordered_stations = sorted(arrivals.keys(), key=lambda k: arrivals[k])

        # Build flattened-trace index map following the data layout
        all_components_list = [component for receiver in self.receivers.iterate() for component in receiver.components]
        time_series_length = int(len(observation) // len(all_components_list))

        trace_index_list = []
        idx = 0
        for receiver in self.receivers.iterate():
            for comp in receiver.components:
                trace_index_list.append((receiver.station_name, comp, idx))
                idx += 1
        trace_index_map = {(st, comp): i for (st, comp, i) in trace_index_list}

        # Reshape observation and ensembles
        obs_matrix = np.reshape(observation, (-1, time_series_length))
        ens_matrices = {
            name: np.reshape(arr, (arr.shape[0], -1, time_series_length))
            for name, arr in ensembles.items()
        }

        # Global normalization based on observation
        max_abs = np.max(np.abs(obs_matrix)) if obs_matrix.size else 1.0
        if max_abs == 0:
            max_abs = 1.0

        components = ['Z', 'E', 'N']
        ordered_by_comp = {
            comp: [(st, comp) for st in ordered_stations if (st, comp) in trace_index_map]
            for comp in components
        }

        # Colors per ensemble
        if colors is None:
            palette = ['blue', 'red', 'plum']
            ens_names = list(ensembles.keys())
            color_map = {name: palette[i % len(palette)] for i, name in enumerate(ens_names)}
        elif isinstance(colors, dict):
            color_map = colors
        else:
            ens_names = list(ensembles.keys())
            color_map = {name: colors[i % len(colors)] for i, name in enumerate(ens_names)}

        # Window half-width in samples
        half_M = M // 2

        spacing = 0.9
        height = 0.7 * 8.0
        width = 12.6 * 1.2
        fig, axes = plt.subplots(1, 3, figsize=(width, height))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            n_traces = len(pairs)
            if n_traces == 0:
                ax.axis('off')
                continue

            offsets = np.arange(n_traces) * spacing

            # Plot observation and ensembles for each trace
            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                obs_trace = obs_matrix[flat_idx]
                # find index of max abs amplitude
                peak_idx = int(np.argmax(np.abs(obs_trace)))
                start = max(0, peak_idx - half_M)
                end = min(time_series_length, start + M)
                # adjust start if near end to keep window length M when possible
                if end - start < M:
                    start = max(0, end - M)
                win_obs = obs_trace[start:end] / max_abs

                # local time axis relative to peak (seconds)
                local_len = end - start
                t_local = (np.arange(local_len) - (peak_idx - start)) / float(self.sampling_rate)

                y0 = offsets[k]
                ax.plot(t_local, y0 + win_obs, color='black', linewidth=1.2, zorder=3,
                        label='Observation' if (i == 0 and k == 0) else None)

                # Ensembles
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

            ax.set_yticks(offsets)
            ax.set_yticklabels([st for (st, _) in pairs])
            ax.set_xlabel('Time relative to peak [s]')
            amp_pad = 1.1
            ax.set_ylim([-amp_pad, (offsets[-1] + amp_pad) if n_traces > 0 else amp_pad])
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_title(f'{comp}')

        # Build combined legend from labels set above
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
            if i == 0:
                # keep both x and y labels
                pass
            else:
                ax.tick_params(labelleft=False)
            ax.set_ylabel('')

        plt.tight_layout()
        if figname is not None:
            fig.savefig(figname, dpi=300, transparent=True)
            fig.clear()
        else:
            plt.show()
        plt.close()

    def plot_posterior_predictive_quantile_traces(
        self,
        observation,
        ensembles,
        event_location,
        colors=None,
        quantiles=(0.05, 0.95),
        fill_alpha=0.25,
        line_width=1.2,
        figname=None,
    ):
        """
        Posterior predictive plot with median and 5–95% bands for each ensemble.

        - Stations are ordered by earliest P-arrival (top to bottom).
        - Three columns: Z, E, N. Each row within a column is a station trace.
        - Observation shown in black.
        - For each ensemble: solid line = median, shaded = [q_low, q_high].

        Parameters
        ----------
        observation : np.ndarray
            1D vector of length data_vector_length.
        ensembles : Dict[str, np.ndarray]
            name -> samples [N_samples, data_vector_length].
        event_location : tuple
            (lat, lon, depth_km).
        colors : list|dict|None
            Colors per ensemble (dict) or assigned in order (list).
        quantiles : tuple(float, float)
            Lower/upper quantiles for shading.
        fill_alpha : float
            Alpha for the shaded band.
        line_width : float
            Line width for median lines.
        figname : str|None
            If provided, saves figure; else shows interactively.
        """
        q_low, q_high = quantiles
        arrivals = self._get_arrivals_dict(event_location)
        ordered_stations = sorted(arrivals.keys(), key=lambda k: arrivals[k])

        all_components_list = [c for r in self.receivers.iterate() for c in r.components]
        time_series_length = int(len(observation) // len(all_components_list))

        trace_index_list = []
        idx = 0
        for receiver in self.receivers.iterate():
            for comp in receiver.components:
                trace_index_list.append((receiver.station_name, comp, idx))
                idx += 1
        trace_index_map = {(st, comp): i for (st, comp, i) in trace_index_list}

        obs_matrix = np.reshape(observation, (-1, time_series_length))
        ens_matrices = {
            name: np.reshape(arr, (arr.shape[0], -1, time_series_length))
            for name, arr in ensembles.items()
        }

        max_abs = np.max(np.abs(obs_matrix)) if obs_matrix.size else 1.0
        if max_abs == 0:
            max_abs = 1.0

        components = ['Z', 'E', 'N']
        ordered_by_comp = {
            comp: [(st, comp) for st in ordered_stations if (st, comp) in trace_index_map]
            for comp in components
        }

        t = np.arange(time_series_length) / float(self.sampling_rate)

        if colors is None:
            palette = ['cornflowerblue', 'crimson', 'plum', 'seagreen', 'orange']
            ens_names = list(ensembles.keys())
            color_map = {name: palette[i % len(palette)] for i, name in enumerate(ens_names)}
        elif isinstance(colors, dict):
            color_map = colors
        else:
            ens_names = list(ensembles.keys())
            color_map = {name: colors[i % len(colors)] for i, name in enumerate(ens_names)}

        spacing = 0.9
        height = 0.7 * 8.0
        width = 12.6 * 1.2
        fig, axes = plt.subplots(1, 3, figsize=(width, height))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        for i, comp in enumerate(components):
            ax = axes[i]
            pairs = ordered_by_comp[comp]
            n_traces = len(pairs)
            if n_traces == 0:
                ax.axis('off')
                continue

            offsets = np.arange(n_traces) * spacing

            for k, (st, c) in enumerate(pairs):
                flat_idx = trace_index_map[(st, c)]
                d = obs_matrix[flat_idx] / max_abs
                y0 = offsets[k]
                ax.plot(t, y0 + d, color='black', linewidth=1.2, zorder=3, label='Observation' if (i == 0 and k == 0) else None)

            for name, sample_cube in ens_matrices.items():
                color = color_map.get(name, 'gray')
                if sample_cube.shape[0] == 0:
                    continue
                for k, (st, c) in enumerate(pairs):
                    flat_idx = trace_index_map[(st, c)]
                    y0 = offsets[k]
                    samples = sample_cube[:, flat_idx, :] / max_abs
                    q_lo = np.quantile(samples, q_low, axis=0)
                    med = np.quantile(samples, 0.5, axis=0)
                    q_hi = np.quantile(samples, q_high, axis=0)
                    ax.fill_between(t, y0 + q_lo, y0 + q_hi, color=color, alpha=fill_alpha, linewidth=0, zorder=1)
                    # Added dashed quantile boundary lines
                    ax.plot(t, y0 + q_lo, color=color, linestyle='--', linewidth=line_width, zorder=2)
                    ax.plot(t, y0 + q_hi, color=color, linestyle='--', linewidth=line_width, zorder=2)
                    ax.plot(t, y0 + med, color=color, linewidth=line_width, zorder=3, label=name if (i == 0 and k == 0) else None)

            ax.set_yticks(offsets)
            ax.set_yticklabels([st for (st, _) in pairs])
            ax.set_xlabel('Time [s]')
            amp_pad = 1.1
            ax.set_ylim([-amp_pad, (offsets[-1] + amp_pad) if n_traces > 0 else amp_pad])
            ax.invert_yaxis()
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.grid(False)
            ax.set_title(f'{comp}')

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
            if i == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(labelleft=False)
            ax.set_ylabel('')

        plt.tight_layout()
        if figname is not None:
            fig.savefig(figname, dpi=300, transparent=True)
            fig.clear()
        else:
            plt.show()
        plt.close()

# ...existing code...

# ...existing code...
def plot_stacked_spectrograms(receivers, flattened_seismogram_array, reference_noise_array=None, sampling_rate = 1, figname = None):
    num_receivers = len(receivers)
    station_names = [receiver.station_name for receiver in receivers]
    time_series_length = int(flattened_seismogram_array.shape[0]//num_receivers)
    seismograms = np.reshape(flattened_seismogram_array, (num_receivers,time_series_length))
    if reference_noise_array is not None:
        reference_noise_array = np.reshape(reference_noise_array, (num_receivers,time_series_length))


    fig, axes = plt.subplots(m.ceil(num_receivers/6), 6, figsize=(15, num_receivers//2))
    
    colors = []
    for index, (ax, seismogram, station_name) in enumerate(zip(axes.ravel(), seismograms, station_names)):
        # ax.specgram(seismogram, Fs=sampling_rate, NFFT=64, noverlap=32, cmap="jet")
        if reference_noise_array is None:
            norm = spectrogram(seismogram, sampling_rate, 
                        per_lap=0.9, wlen=20, log=True, dbscale=True, 
                        cmap="jet", show=False, axes=ax)
        else:
            specgram,freq,time, end = compute_spectrogram(seismogram, sampling_rate, per_lap=0.9, wlen=20, dbscale=False)
            specgram_noise, _, _, _ = compute_spectrogram(reference_noise_array[index], sampling_rate, per_lap=0.9, wlen=20, dbscale=False)
            norm = generate_spectrogram_plot(10 * np.log10(specgram / specgram_noise), freq, time, end, log=True, axes=ax, cmap="jet")
        
        ax.set_title(f"{station_name} - Peak: {int(norm.vmax)} dB/Hz")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")


    for i in range(num_receivers, len(axes.ravel())):
        axes.ravel()[i].axis("off")
    plt.tight_layout()
    if figname is not None:
        fig.savefig(figname)
        fig.clear()
    else:
        plt.show()
    plt.close()
    return colors

import math

import numpy as np
from matplotlib import mlab
from matplotlib.colors import Normalize

from obspy.imaging.cm import obspy_sequential


def _nearest_pow_2(x):
    """
    Find power of two nearest to x

    >>> _nearest_pow_2(3)
    2.0
    >>> _nearest_pow_2(15)
    16.0

    :type x: float
    :param x: Number
    :rtype: Int
    :return: Nearest power of 2 to x
    """
    a = math.pow(2, math.ceil(np.log2(x)))
    b = math.pow(2, math.floor(np.log2(x)))
    if abs(a - x) < abs(b - x):
        return a
    else:
        return b


def spectrogram(data, samp_rate, per_lap=0.9, wlen=None, log=False,
                outfile=None, fmt=None, axes=None, dbscale=False,
                mult=8.0, cmap=obspy_sequential, zorder=None, title=None,
                show=True, sphinx=False, clip=[0.0, 1.0]):
    """
    Computes and plots spectrogram of the input data.

    :param data: Input data
    :type samp_rate: float
    :param samp_rate: Samplerate in Hz
    :type per_lap: float
    :param per_lap: Percentage of overlap of sliding window, ranging from 0
        to 1. High overlaps take a long time to compute.
    :type wlen: int or float
    :param wlen: Window length for fft in seconds. If this parameter is too
        small, the calculation will take forever. If None, it defaults to
        (samp_rate/100.0).
    :type log: bool
    :param log: Logarithmic frequency axis if True, linear frequency axis
        otherwise.
    :type outfile: str
    :param outfile: String for the filename of output file, if None
        interactive plotting is activated.
    :type fmt: str
    :param fmt: Format of image to save
    :type axes: :class:`matplotlib.axes.Axes`
    :param axes: Plot into given axes, this deactivates the fmt and
        outfile option.
    :type dbscale: bool
    :param dbscale: If True 10 * log10 of color values is taken, if False the
        sqrt is taken.
    :type mult: float
    :param mult: Pad zeros to length mult * wlen. This will make the
        spectrogram smoother.
    :type cmap: :class:`matplotlib.colors.Colormap`
    :param cmap: Specify a custom colormap instance. If not specified, then the
        default ObsPy sequential colormap is used.
    :type zorder: float
    :param zorder: Specify the zorder of the plot. Only of importance if other
        plots in the same axes are executed.
    :type title: str
    :param title: Set the plot title
    :type show: bool
    :param show: Do not call `plt.show()` at end of routine. That way, further
        modifications can be done to the figure before showing it.
    :type sphinx: bool
    :param sphinx: Internal flag used for API doc generation, default False
    :type clip: [float, float]
    :param clip: adjust colormap to clip at lower and/or upper end. The given
        percentages of the amplitude range (linear or logarithmic depending
        on option `dbscale`) are clipped.
    """

    # enforce float for samp_rate
    specgram, freq, time, end = compute_spectrogram(data, samp_rate, per_lap, wlen, dbscale, mult)

    norm = generate_spectrogram_plot(specgram, freq, time, end, log, axes,  cmap, zorder, clip)

    return norm

def generate_spectrogram_plot(specgram, freq, time, end, log=False, axes=None,  cmap=obspy_sequential, zorder=None, clip=[0.0, 1.0]):
    import matplotlib.pyplot as plt
    vmin, vmax = clip
    if vmin < 0 or vmax > 1 or vmin >= vmax:
        msg = "Invalid parameters for clip option."
        raise ValueError(msg)
    _range = float(specgram.max() - specgram.min())
    vmin = specgram.min() + vmin * _range
    vmax = specgram.min() + vmax * _range
    norm = Normalize(vmin, vmax, clip=True)

    if not axes:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = axes

    # calculate half bin width
    halfbin_time = (time[1] - time[0]) / 2.0
    halfbin_freq = (freq[1] - freq[0]) / 2.0

    # argument None is not allowed for kwargs on matplotlib python 3.3
    kwargs = {k: v for k, v in (('cmap', cmap), ('zorder', zorder))
              if v is not None}

    if log:
        # pcolor expects one bin more at the right end
        freq = np.concatenate((freq, [freq[-1] + 2 * halfbin_freq]))
        time = np.concatenate((time, [time[-1] + 2 * halfbin_time]))
        # center bin
        time -= halfbin_time
        freq -= halfbin_freq
        # Log scaling for frequency values (y-axis)
        ax.set_yscale('log')
        # Plot times
        ax.pcolormesh(time, freq, specgram, norm=norm, **kwargs)
    else:
        # this method is much much faster!
        specgram = np.flipud(specgram)
        # center bin
        extent = (time[0] - halfbin_time, time[-1] + halfbin_time,
                  freq[0] - halfbin_freq, freq[-1] + halfbin_freq)
        ax.imshow(specgram, interpolation="nearest", extent=extent, **kwargs)

    # set correct way of axis, whitespace before and after with window
    # length
    ax.axis('tight')
    ax.set_xlim(0, end)
    ax.grid(False)
    return norm

def compute_spectrogram(data, samp_rate, per_lap=0.9, wlen=None, dbscale=False,mult=8.0,):
    samp_rate = float(samp_rate)

    # set wlen from samp_rate if not specified otherwise
    if not wlen:
        wlen = samp_rate / 100.

    npts = len(data)
    # nfft needs to be an integer, otherwise a deprecation will be raised
    # XXX add condition for too many windows => calculation takes for ever
    nfft = int(_nearest_pow_2(wlen * samp_rate))
    if nfft > npts:
        nfft = int(_nearest_pow_2(npts / 8.0))

    if mult is not None:
        mult = int(_nearest_pow_2(mult))
        mult = mult * nfft
    nlap = int(nfft * float(per_lap))

    data = data - data.mean()
    end = npts / samp_rate

    # Here we call not plt.specgram as this already produces a plot
    # matplotlib.mlab.specgram should be faster as it computes only the
    # arrays
    # XXX mlab.specgram uses fft, would be better and faster use rfft
    specgram, freq, time = mlab.specgram(data, Fs=samp_rate, NFFT=nfft,
                                         pad_to=mult, noverlap=nlap)
    # db scale and remove zero/offset for amplitude
    if dbscale:
        specgram = 10 * np.log10(specgram[1:, :])
    else:
        specgram = np.sqrt(specgram[1:, :])
    freq = freq[1:]
    return specgram,freq,time, end


