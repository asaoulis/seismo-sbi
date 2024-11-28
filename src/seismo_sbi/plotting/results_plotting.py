from pathlib import Path
import numpy as np
from sbi import utils as utils
from sbi import analysis as analysis


from seismo_sbi.instaseis_simulator.receivers import Receivers
from seismo_sbi.plotting.seismo_plots import plot_stacked_waveforms, MisfitsPlotting
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

        self.posterior_plotter = PosteriorPlotter(data_scaler, parameters_info, self.parameters)

        if "moment_tensor" in self.parameters.names.keys():
            self.reparametrised_plotter = MomentTensorReparametrised(data_scaler, self.parameters)

    def plot_all_stacked_waveforms(self, single_job : JobData, receivers : Receivers):

        figure_path = self.base_output_path / "./seismograms"
        figure_path.mkdir(parents=True, exist_ok=True)

        plot_path = figure_path / f"./{single_job.job_name}.png"
        plot_stacked_waveforms(receivers.receivers, single_job.data_vector, figname=plot_path)
    
    def plot_synthetic_misfits(self, single_job : JobData, receivers : Receivers, synthetics : np.ndarray, event_location, covariance = None, only_raw=False, savefig=True):
            
        figure_path = self.base_output_path / "./misfits"
        figure_path.mkdir(parents=True, exist_ok=True)

        misfits_plotter = MisfitsPlotting(receivers, 1, covariance)
        data_vector = single_job.data_vector

        plot_path = figure_path / f"./raw_{single_job.job_name}.png" if savefig else None
        misfits_plotter.raw_synthetic_misfits(data_vector, synthetics, figname=plot_path)
        try:
            if not only_raw:
                plot_path = figure_path / f"./arrival_{single_job.job_name}.png" if savefig else None
                misfits_plotter.arrival_synthetic_misfits(data_vector, synthetics, (*event_location, 20), figname=plot_path)
        except:
            pass

    
    def plot_posterior(self, test_name, inversion_data, kde=True, savefig=True):

        figure_path = self.base_output_path / "./inversions" if savefig else None
        if savefig:
            figure_path.mkdir(parents=True, exist_ok=True)

        self.plot_chain_consumer(f"inversions", test_name, {"":inversion_data}, kde=kde, savefig=savefig)

        if "moment_tensor" in self.parameters.names.keys():
            plot_path = self.base_output_path / f"./beachballs/{test_name}"  if savefig else None
            if savefig:
                plot_path.parent.mkdir(parents=True, exist_ok=True)
            self.posterior_plotter.plot_beachball_samples(inversion_data, plot_path=plot_path)

    def plot_chain_consumer(self, base_figure_path, test_name, inversion_data_dict, kde=True, savefig=True):
        plot_path = self.base_output_path / f"./{base_figure_path}" / f"./{test_name}.png"  if savefig else None
        if savefig:
            plot_path.parent.mkdir(parents=True, exist_ok=True)

        self.posterior_plotter.plot_chain_consumer(inversion_data_dict, kde=kde, figsave=plot_path)

        if "moment_tensor" in self.parameters.names.keys():
            plot_path = self.base_output_path / f"./{base_figure_path}" / f"./nodal_params_{test_name}.png"  if savefig else None
            self.reparametrised_plotter.plot_chain_consumer(inversion_data_dict, kde=kde, inverse=True, figsave=plot_path)

    def plot_compression(self, raw_compressed_dataset, compressed_estimate = None, job_name=None):
        plotting_base_output_path = self.base_output_path
        if job_name is not None:
            plotting_base_output_path = plotting_base_output_path / 'compression'
        plotting_base_output_path.mkdir(exist_ok=True, parents=True)

        figname = (plotting_base_output_path / f"./{job_name}.png").resolve()
        self.posterior_plotter.plot_compression_errors(raw_compressed_dataset[:, :2*self.num_dim], compressed_estimate, figname=figname)
    
