"""Per-job figures produced by the inference pipeline.

:class:`SBIPipelinePlotter` writes the stacked waveforms, synthetic misfits, corner plots and
compression diagnostics for one job under the pipeline's output directory.
"""

from pathlib import Path
import numpy as np


from seismo_sbi.simulators.receivers import Receivers
from seismo_sbi.plotting.seismo_plots import MisfitsPlotting
from seismo_sbi.plotting.distributions import PosteriorPlotter, MomentTensorReparametrised
from seismo_sbi.sbi.configuration import  ModelParameters
from seismo_sbi.sbi.types.results import JobData

class SBIPipelinePlotter:

    def __init__(self, base_output_path, parameters : ModelParameters):

        self.base_output_path = Path(base_output_path)
        self.parameters = parameters
        self.num_dim = parameters.parameter_to_vector('theta_fiducial').shape[0]

        self.posterior_plotter = None
        self.reparametrised_plotter = None

    def initialise_posterior_plotter(self, data_scaler, parameters_info):
        """Build the posterior plotters, and the moment-tensor one when the run infers a tensor."""

        self.posterior_plotter = PosteriorPlotter(data_scaler, parameters_info, self.parameters)

        if "moment_tensor" in self.parameters.names.keys():
            self.reparametrised_plotter = MomentTensorReparametrised(data_scaler, self.parameters)

    def plot_synthetic_misfits(self, single_job : JobData, receivers : Receivers, synthetics : np.ndarray, event_location, sampling_rate_hz, covariance = None, only_raw=False, savefig=True, vertical_only=False, time_window_s=None):
        """Observed against synthetic traces of one job; ``vertical_only`` draws the vertical traces
        alone, ordered by P arrival, in one figure. ``event_location`` is ``(latitude, longitude,
        depth_km)`` of the source and ``sampling_rate_hz`` the traces' sampling rate. ``time_window_s``
        = (start, end) in seconds limits the stacked-trace figure to that part of the window."""
        figure_path = self.base_output_path / "./misfits"
        figure_path.mkdir(parents=True, exist_ok=True)

        misfits_plotter = MisfitsPlotting(receivers, sampling_rate_hz, covariance)
        data_vector = single_job.data_vector
        if vertical_only:
            misfits_plotter.plot_ordered_stacked_traces(data_vector, synthetics, event_location, figname=f"{figure_path}/stacked_{single_job.job_name}.png" if savefig else None, components=('Z',), time_window_s=time_window_s)
            return

        plot_path = figure_path / f"./raw_{single_job.job_name}.png" if savefig else None
        misfits_plotter.raw_synthetic_misfits(data_vector, synthetics, figname=plot_path)
        print("Plotting raw misfits... ordered stacked traces.")
        misfits_plotter.plot_ordered_stacked_traces(data_vector, synthetics, event_location, figname=f"{figure_path}/stacked_{single_job.job_name}.png" if savefig else None, time_window_s=time_window_s)
        if not only_raw:
            plot_path = figure_path / f"./arrival_{single_job.job_name}.png" if savefig else None
            misfits_plotter.arrival_synthetic_misfits(data_vector, synthetics, event_location, figname=plot_path)

    
    def plot_posterior(self, test_name, inversion_data, kde=True, savefig=True):

        figure_path = self.base_output_path / "./inversions" if savefig else None
        if savefig:
            figure_path.mkdir(parents=True, exist_ok=True)

        self.plot_chain_consumer("inversions", test_name, {"":inversion_data}, kde=kde, savefig=savefig)

        if "moment_tensor" in self.parameters.names.keys():
            plot_path = self.base_output_path / f"./beachballs/{test_name}"  if savefig else None
            if savefig:
                plot_path.parent.mkdir(parents=True, exist_ok=True)
            self.posterior_plotter.plot_beachball_samples(inversion_data, plot_path=plot_path)

    def plot_chain_consumer(self, base_figure_path, test_name, inversion_data_dict, kde=True, savefig=True, **kwargs):
        """Corner plots (and lunes, for a tensor) of one job's inversions, under ``base_figure_path``."""
        lune_kwargs, reparam_kwargs = kwargs.get("lune_kwargs", {}), kwargs.get("reparam_kwargs", {})
        # A relative base_figure_path nests under base_output_path; an absolute one is used as is.
        figure_dir = self.base_output_path / base_figure_path
        plot_path = figure_dir / f"{test_name}.svg" if savefig else None
        if savefig:
            figure_dir.mkdir(parents=True, exist_ok=True)
        print("Added lune kwargs")

        self.posterior_plotter.plot_chain_consumer(inversion_data_dict, kde=kde, figsave=plot_path)
        if "moment_tensor" in self.parameters.names.keys():
            # Scatter lune plot
            plot_path = figure_dir / f"lune_{test_name}.svg" if savefig else None
            self.posterior_plotter.plot_lunes(inversion_data_dict, figsave=plot_path, **lune_kwargs)
            # KDE lune plot
            plot_kde_path = figure_dir / f"lune_kde_{test_name}.svg" if savefig else None
            self.posterior_plotter.plot_lunes_kde(inversion_data_dict, figsave=plot_kde_path, **lune_kwargs)
            # Nodal parameter corner plot
            plot_path = figure_dir / f"nodal_params_{test_name}.svg" if savefig else None
            self.reparametrised_plotter.plot_chain_consumer(inversion_data_dict, kde=kde, inverse=False, figsave=plot_path, **reparam_kwargs)

    def plot_compression(self, raw_compressed_dataset, compressed_estimate = None, job_name=None):
        plotting_base_output_path = self.base_output_path
        figname = None
        if job_name:
            plotting_base_output_path = plotting_base_output_path / 'compression'
            figname = (plotting_base_output_path / f"./{job_name}.png").resolve()
            plotting_base_output_path.mkdir(exist_ok=True, parents=True)

        self.posterior_plotter.plot_compression_errors(raw_compressed_dataset[:, :2*self.num_dim], compressed_estimate, figname=figname)

