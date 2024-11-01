
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from .parameters import ParameterInformation
import torch
from chainconsumer import ChainConsumer
from obspy.imaging.beachball import beach
from obspy.imaging import beachball
from pyrocko.plot import beachball as rocko_beachball
import pyrocko.moment_tensor as mtm
from ..instaseis_simulator.dataset_generator import tqdm_joblib
from tqdm import tqdm
from contextlib import contextmanager
import logging

# disable chain consumer warnings for reparametrised moment tensor
# as the angle distributions cover periodic sample space
# solution from https://gist.github.com/simon-weber/7853144
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


def transfer_labels_and_ticks(src_ax, dest_ax):
    # Transfer x-axis labels and ticks
    dest_ax.set_xlabel(src_ax.get_xlabel())
    dest_ax.set_xticks(src_ax.get_xticks())
    dest_ax.set_xticklabels(src_ax.get_xticklabels())
    dest_ax.xaxis.set_ticks_position(src_ax.xaxis.get_ticks_position())
    
    # Transfer y-axis labels and ticks
    dest_ax.set_ylabel(src_ax.get_ylabel())
    dest_ax.set_yticks(src_ax.get_yticks())
    dest_ax.set_yticklabels(src_ax.get_yticklabels())
    dest_ax.yaxis.set_ticks_position(src_ax.yaxis.get_ticks_position())

def flip_subplots(fig, axes):
    n = len(axes)
    
    # Store the original positions of the subplots that need to be moved
    positions = {}
    for i in range(n):
        for j in range(n):
            if i > j:
                positions[(i, j)] = axes[i, j].get_position()
    for i in range(n):
        transfer_labels_and_ticks(axes[0, i], axes[i, 0]) # Temporarily set the original axis to the new position
        transfer_labels_and_ticks(axes[i, 0], axes[0, i])

    # Move the subplots to the new positions
    for (i, j), pos in positions.items():
        new_i, new_j = j, i
        axes[i, j].set_position(axes[new_i, new_j].get_position())  # Temporarily set the original axis to the new position
        axes[new_i, new_j].set_position(pos)  # Move the target axis to the original position

    for i in range(n):
        transfer_labels_and_ticks(axes[0, i], axes[i, 0]) # Temporarily set the original axis to the new position
        transfer_labels_and_ticks(axes[i, 0], axes[0, i])


        
    # Clean up the moved axes
    for i in range(n):
        for j in range(n):
            if i < j:
                axes[i, j].remove()

    return fig, axes

class DummyDataScaler:

    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_

    def inverse_transform(self, data):
        return data
    
    def transform(self, data):
        return data

def get_MW_and_epsilon(moment_tensor_sol):

    moment_tensor_matrix, M_0 = compute_scalar_moment(moment_tensor_sol)
    MW = (np.log10(M_0) - 9.1)/1.5

    M_isotropic = 1/3 * np.trace(moment_tensor_matrix) * np.eye(3)
    M_deviatoric = moment_tensor_matrix - M_isotropic

    eigenvalues = list(sorted(np.linalg.eigvals(M_deviatoric), reverse=True))
    epsilon = eigenvalues[1]/ max(abs(eigenvalues[0]), abs(eigenvalues[2]))
    
    return (MW, epsilon)

def compute_scalar_moment(moment_tensor_sol):
    moment_tensor_matrix = create_matrix(moment_tensor_sol)
    
    M_0 = (1/np.sqrt(2)) * np.sum(moment_tensor_matrix**2)**(1/2)
    return moment_tensor_matrix, M_0

def create_matrix(moment_tensor_sol):
    moment_tensor_matrix = np.array([[moment_tensor_sol[0], moment_tensor_sol[3], moment_tensor_sol[4]],
                                        [moment_tensor_sol[3], moment_tensor_sol[1], moment_tensor_sol[5]],
                                        [moment_tensor_sol[4], moment_tensor_sol[5], moment_tensor_sol[2]]])
                                        
    return moment_tensor_matrix
from pyrocko import moment_tensor as pmt

def convert_to_pyrocko(mt):
    #up, south, east to north east down
    m = pmt.MomentTensor(
        mnn=-mt[1],
        mee=-mt[2],
        mdd=-mt[0],
        mne=-mt[4],
        mnd=-mt[5],
        med=-mt[3]
    )
    return m

def get_nodal_planes(theta):
    m = convert_to_pyrocko(theta)
    nodal_planes = m.both_strike_dip_rake()
    return nodal_planes

class MomentTensorReparametrised:

    parameters_info = [ParameterInformation("$strike$", '°'),
                        ParameterInformation("$dip$", '°'),
                        ParameterInformation("$rake$", '°'),
                        ParameterInformation("$M_w$", ""),
                        ParameterInformation("$\epsilon$", "")]
    
    def __init__(self, data_scaler, parameters):
        # to convert samples from nde training scale
        # back to natural units
        self.data_scaler = data_scaler
        self.parameters = parameters
        # no need for an extra scaling step
        dummy_scaler = DummyDataScaler(5)
        self.chain_plotter = PosteriorPlotter(dummy_scaler, self.parameters_info)

    def convert_samples(self, samples, theta0, custom_select):
        
        nodal_planes = []
        for sample in samples:
            inputs = self.parameters.vector_to_simulation_inputs(sample, only_theta_fiducial=True)
            sample = inputs["moment_tensor"]
            magnitude, epsilon = get_MW_and_epsilon(sample)
            nodal_plane_pair = get_nodal_planes(sample)
            nodal_plane = nodal_plane_pair[0] if not custom_select else custom_select(nodal_plane_pair)
            nodal_planes.append(np.array([nodal_plane[0], nodal_plane[1], nodal_plane[2], magnitude, epsilon]))
        if theta0 is not None:
            theta_inputs = self.parameters.vector_to_simulation_inputs(theta0, only_theta_fiducial=True)
            theta0 = theta_inputs["moment_tensor"]
            theta_MW, theta_epsilon = get_MW_and_epsilon(theta0)
            nodal_plane_pair = get_nodal_planes(theta0)
            nodal_plane = nodal_plane_pair[0] if not custom_select else custom_select(nodal_plane_pair)
            theta0_converted = np.array([nodal_plane[0], nodal_plane[1], nodal_plane[2], theta_MW, theta_epsilon])
        else:
            theta0_converted = None
        return np.array(nodal_planes), theta0_converted



    def plot_chain_consumer(self, samples_theta0_dict, custom_processing = None, *args, **kwargs):
        converted_chain_dict = self.convert_chain_dict(samples_theta0_dict, custom_processing)

        with warning_logging_disabled():
            self.chain_plotter.plot_chain_consumer(converted_chain_dict,  *args, **kwargs)

    def convert_chain_dict(self, samples_theta0_dict, custom_processing):
        converted_chain_dict = {}
        for name, (theta0, samples, data_scaler, _) in samples_theta0_dict.items():
            if data_scaler is None:
                data_scaler = self.data_scaler
            samples = data_scaler.inverse_transform(samples)
            if theta0 is not None:
                theta0 = self.data_scaler.inverse_transform(theta0.reshape(1, -1)).flatten()
            samples, theta0 = self.convert_samples(samples, theta0, custom_processing)
            # if custom_processing is not None:
            #     samples, theta0 = custom_processing(samples, theta0)
            converted_chain_dict[name] = (theta0, samples, None)
        return converted_chain_dict


class PosteriorPlotter:

    def __init__(self, data_scaler, parameters_info : List[ParameterInformation], parameters = None, num_jobs = 0):

        self.data_scaler = data_scaler
        self.parameters_info = parameters_info
        self.parameters = parameters
        self.num_dim = len(parameters_info)
        self.num_jobs = num_jobs

    def plot_compression_likelihood(self, compressed_dataset, parameter_index, likelihood_estimator, figname=None):

        ground_truths = compressed_dataset[:, :self.num_dim]
        compressions = compressed_dataset[:, self.num_dim:]
        ground_truths = self._transform_to_plotting_units(ground_truths)
        compressions = self._transform_to_plotting_units(compressions)
        

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,12))
        ax.axis("off")
        parameter = self.parameters_info[parameter_index]
        param_ground_truths = ground_truths[:, parameter_index]
        param_compressions = compressions[:, parameter_index]
        
        ax.set_title(f"{parameter.name}")
        ax.scatter(param_ground_truths, param_compressions, label="Compression", marker='x', alpha=0.5)
        # ax.plot(param_ground_truths, param_ground_truths, label="Ground truth", color="green", linestyle='--')
        ax.set_xlabel(f"Ground truth ({parameter.unit})")
        ax.set_ylabel(f"Compression ({parameter.unit})")
        
        xs = np.linspace(0,1, 20)
        
        xx, yy = np.meshgrid(xs, xs)
        xx = np.expand_dims(xx, -1)
        yy =np.expand_dims(yy, -1)
        base_vector= 0.5* np.ones((self.num_dim - 1))
        repeated = np.tile(base_vector, (20, 20, 1))
        thetas = torch.Tensor(np.concatenate([repeated[:,:,:parameter_index], xx, repeated[:,:,parameter_index:]], axis=-1)).flatten(start_dim=0, end_dim=1)
        compressions  = torch.Tensor(np.concatenate([repeated[:,:,:parameter_index], yy, repeated[:,:,parameter_index:]], axis=-1)).flatten(start_dim=0, end_dim=1)
        
        probabilities = likelihood_estimator.log_prob(thetas, compressions)
        # xx, yy = np.meshgrid(real_units_xs, real_units_xs)
        u_thetas = self.data_scaler.inverse_transform(thetas)
        u_compressions = self.data_scaler.inverse_transform(compressions)
        # print(parameter.scaling_transform(u_thetas[:200, parameter_index]),
        #         parameter.scaling_transform(u_compressions[:10, parameter_index]))
        contours = ax.contourf(parameter.scaling_transform(u_thetas[:,parameter_index]).reshape(20,20),
                    parameter.scaling_transform(u_compressions[:, parameter_index].reshape(20,20)), probabilities.reshape(20,20).detach().numpy().T
                    # ,alpha=0.3, levels=[-5000,-2000,-1000,-500,-200,-100,0, 1000])
                    ,alpha=0.3, levels=[-200,-100,-50, -25, -10,0, 200])
        
        plt.tight_layout()

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close()

        plt.hist(probabilities.detach().numpy())
        plt.show()
    def compute_posterior_probabilities(self, compressed_dataset, parameter_index,  likelihood_estimator, resolution=20):

        ground_truths = compressed_dataset[:, :self.num_dim]
        compressions = compressed_dataset[:, self.num_dim:]
        ground_truths = self._transform_to_plotting_units(ground_truths)
        compressions = self._transform_to_plotting_units(compressions)
        
        xs = np.linspace(0,1, resolution)
        
        xx, yy = np.meshgrid(xs, xs)
        xx = np.expand_dims(xx, -1)
        yy =np.expand_dims(yy, -1)
        base_vector= 0.5* np.ones((self.num_dim - 1))
        repeated = np.tile(base_vector, (resolution, resolution, 1))
        thetas = torch.Tensor(np.concatenate([repeated[:,:,:parameter_index], xx, repeated[:,:,parameter_index:]], axis=-1)).flatten(start_dim=0, end_dim=1)
        compressions  = torch.Tensor(np.concatenate([repeated[:,:,:parameter_index], yy, repeated[:,:,parameter_index:]], axis=-1)).flatten(start_dim=0, end_dim=1)
        probabilities = []
        from tqdm import tqdm
        
        if self.num_jobs == 0:
            for theta, compression in zip(thetas, compressions):
                probability = likelihood_estimator.log_prob(theta, compression)
                probabilities.append(probability)
        else:
             with tqdm_joblib(tqdm(desc="Running simulations: ", total=len(thetas))) as progress_bar:
                with joblib.parallel_backend('loky', n_jobs=self.num_jobs):
                    probabilities = joblib.Parallel()(
                        joblib.delayed(likelihood_estimator.log_prob)(theta, compression) for theta, compression in zip(thetas, compressions)
                    )
        flat_observations = np.linspace(0,1, 200)
        compression_vals = [0.3, 0.7]
        posterior_lines = []
        cust_compression_vals = []
        for compression_val in compression_vals:
            cust_thetas = torch.Tensor(0.5*np.ones((200, 6)))
            cust_thetas[:, parameter_index] = torch.Tensor(flat_observations)
            cust_theta_saved = self.data_scaler.inverse_transform(cust_thetas)

            cust_compressions = torch.Tensor(0.5*np.ones((200, 6)))
            cust_compressions[:, parameter_index] = torch.Tensor(np.full_like(flat_observations, compression_val))
            cust_compressions_saved = self.data_scaler.inverse_transform(cust_compressions)
            cust_compression_vals.append(cust_compressions_saved)
            # compressions = torch.Tensor(np.full_like(flat_observations, compression_val))
            with tqdm_joblib(tqdm(desc="Running simulations: ", total=len(cust_thetas))) as progress_bar:
                with joblib.parallel_backend('loky', n_jobs=self.num_jobs):
                    posterior_vals = joblib.Parallel()(
                        joblib.delayed(likelihood_estimator.log_prob)(theta, compression) for theta, compression in zip(cust_thetas, cust_compressions)
                    )
            posterior_lines.append(torch.stack(posterior_vals).numpy())
        
        probabilities = torch.stack(probabilities)
        # xx, yy = np.meshgrid(real_units_xs, real_units_xs)
        u_thetas = self.data_scaler.inverse_transform(thetas)
        u_compressions = self.data_scaler.inverse_transform(compressions)
        return u_thetas, u_compressions, probabilities, (cust_compression_vals, cust_theta_saved, posterior_lines)
    

    def plot_compression_posterior(self, compressed_dataset, parameter_index, likelihood_estimator, figname=None):

        u_thetas, u_compressions, probabilities, posterior_lines = self.compute_posterior_probabilities(compressed_dataset, parameter_index, likelihood_estimator)
        ground_truths = compressed_dataset[:, :self.num_dim]
        compressions = compressed_dataset[:, self.num_dim:]
        ground_truths = self._transform_to_plotting_units(ground_truths)
        compressions = self._transform_to_plotting_units(compressions)
        
        parameter = self.parameters_info[parameter_index]
        param_ground_truths = ground_truths[:, parameter_index]
        param_compressions = compressions[:, parameter_index]
        
        transform = lambda x: x - np.min(param_ground_truths)/ (np.max(param_ground_truths) - np.min(param_ground_truths))
        #width aspect 2:1 
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6,5), sharex=True, gridspec_kw={'height_ratios': [3, 2]})
        ax =axes[0]
        post_ax = axes[1]
        post_ax.set_xticks([])
        post_ax.set_yticks([])
        # post_ax.set_title("Posterior $ p(\\mathbf{m} \mid \\mathbf{D})$")
        post_ax.set_xlabel("Model Parameters, $\\mathbf{m}$")

        # ax.axis("off")
        # ax.set_title(f"Empirical Density Modelling")
        ax.scatter(param_ground_truths, param_compressions, label="Compression", marker='x', alpha=0.7, color='red')
        # ax.plot(param_ground_truths, param_ground_truths, label="Ground truth", color="green", linestyle='--')
        # ax.set_xlabel("Model Parameters, $\\mathbf{m}$")
        # ax.set_ylabel("Observation, $\\mathbf{D}$")
        ax.set_ylim(np.min(parameter.scaling_transform(u_thetas[:,parameter_index])), 
                    np.max(parameter.scaling_transform(u_thetas[:, parameter_index])))
        # print(param_ground_truths)
        # print(parameter.scaling_transform(u_thetas[:200, parameter_index]),
        #         parameter.scaling_transform(u_compressions[:10, parameter_index]))
        shaped_probs = probabilities.reshape(20,20).detach().numpy().T
        contours = ax.contourf(parameter.scaling_transform(u_thetas[:,parameter_index]).reshape(20,20),
                    parameter.scaling_transform(u_compressions[:, parameter_index].reshape(20,20)), np.clip(shaped_probs,-700,10000),
                    alpha=0.4, levels=[ -750,-500, -350, -200, -100, -50, -25,0,50])
                    # levels=[  -200, -150, -125, -100, -75,-50,-35, -20, -10, 0,20])
                    # ,alpha=0.4, levels=[-4, -3, -2.5, -2.2, -1.8, -1.6,  -1,-0.1, 0, 1])

        xs = parameter.scaling_transform(posterior_lines[1][:,parameter_index])
        posterior = np.exp(0.05*np.array(posterior_lines[2][1]))
        posterior /= np.max(posterior) * 0.2
        ys =  parameter.scaling_transform(posterior_lines[0][1][0,parameter_index])* np.ones_like(xs)
        posterior_line = ax.plot(xs, ys, color='blue', label='Posterior', linestyle='--', linewidth=2)
        # post_ax.plot(xs, posterior, color='black', label='Posterior', linestyle='--')
        post_ax.fill_between(xs, posterior.flatten(), alpha=0.6, color='cornflowerblue')
        
        posterior = np.exp(0.05*np.array(posterior_lines[2][0]))
        posterior /= np.max(posterior) * 0.2
        ys =parameter.scaling_transform(posterior_lines[0][0][0,parameter_index]) * np.ones_like(xs)
        posterior_line = ax.plot(xs, ys, color='red', label='Posterior', linestyle='--', linewidth=2)
        # post_ax.plot(xs, posterior, color='black', label='Posterior', linestyle='--')
        post_ax.fill_between(xs, posterior.flatten(), alpha=0.6, color='red')
        post_ax.set_ylim(0.001, np.max(posterior.flatten()) * 1.2)

        ax.set_xticks([])
        ax.set_yticks([parameter.scaling_transform(posterior_lines[0][0][0,parameter_index]), parameter.scaling_transform(posterior_lines[0][1][0,parameter_index])])
        ax.set_yticklabels(["$\\mathbf{D}_1$", '$\\mathbf{D}_2$'])
        plt.tight_layout()

        if figname is not None:
            fig.savefig(figname, dpi=200, transparent=True)
            fig.clear()
        else:
            plt.show()
        plt.close()

    
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
            # if i == 0:
            #     left_ax.legend()
            
        plt.tight_layout()

        if figname is not None:
            fig.savefig(figname)
            fig.clear()
        else:
            plt.show()
        plt.close()

    def plot_chain_consumer(self, inversion_data, kde=True, extents=None, inverse=False, figsave= None, tick_font_size=30, *args, **kwargs):

        colors = ['blue', 'red', 'purple', 'green', 'brown']

        scaled_data_dict = {name: self._prepare_data_for_plotting(*data) 
                                for name, data in inversion_data.items()}

        parameters_label = [f"{parameters_info.name} [{parameters_info.unit}]" if parameters_info.unit != "" 
                                    else parameters_info.name for parameters_info in self.parameters_info]
                            
        i = 0
        c_plot = ChainConsumer()
        for name, (samples, theta0) in scaled_data_dict.items():
            if extents is not None:
                for j in range(len(extents)):
                    samples = samples[samples[:, j] > extents[j][0]]
                    samples = samples[samples[:, j] < extents[j][1]]

            if i == 0:
                truth = theta0
                shade = True
            else:
                shade = False
            c_plot.add_chain(samples, parameters=parameters_label, color=colors[i], name=name, shade=shade, linewidth=2.5)
            i+=1
        c_plot.configure(kde=[kde for _ in range(len(inversion_data))], shade_alpha=0.7, max_ticks=3, diagonal_tick_labels=False, tick_font_size=tick_font_size, label_font_size=40, summary=True, usetex=True, bar_shade=True)
        c_plot.configure_truth(lw=2)
        scale = 3*self.num_dim

        fig = c_plot.plotter.plot(figsize=(scale,scale), truth=truth, legend=False, extents=extents)
        fig.align_labels() 

        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight")
        plt.close()

    def plot_posterior_distribution(self, samples, theta0, bounds, figsave= None):


        plotting_units_samples, plotting_units_theta_0 = self._prepare_data_for_plotting(theta0, samples)
        
        fig, axes = plt.subplots(self.num_dim, 2*self.num_dim,
                                    figsize = (8*self.num_dim,4 * self.num_dim))

        bounds_array = np.vstack(bounds)
        plotting_bounds = self._transform_to_plotting_units(bounds_array)

        full_prior_plot_axes = axes[:, :self.num_dim:]
        zoom_plot_axes = axes[:, self.num_dim:]
        try:
            self._add_triangle_plot_to_axes(plotting_units_samples,
                                            plotting_units_theta_0,
                                            plotting_bounds,
                                            full_prior_plot_axes)
        except:
            pass

        self._add_triangle_plot_to_axes(plotting_units_samples,
                                        plotting_units_theta_0,
                                        None,
                                        zoom_plot_axes)
        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave)
        plt.close()

    def _prepare_data_for_plotting(self, theta0, samples, data_scaler = None, *args, **kwargs):

        if data_scaler is None:
            data_scaler = self.data_scaler
        raw_units_samples = data_scaler.inverse_transform(samples)
        if theta0 is not None:
            raw_units_theta_0 = data_scaler.inverse_transform(theta0.reshape(1, -1))

        plotting_units_samples = self._transform_to_plotting_units(raw_units_samples)
        if theta0 is not None:
            plotting_units_theta_0 = self._transform_to_plotting_units(raw_units_theta_0).flatten()
        else:
            plotting_units_theta_0 = None
        return plotting_units_samples,plotting_units_theta_0

    def _transform_to_plotting_units(self, samples):
        plotting_units_samples = np.copy(samples)
        for i in range(samples.shape[1]):
            scaler = self.parameters_info[i].scaling_transform
            plotting_units_samples[:,i] = scaler(samples[:,i])
        return plotting_units_samples

    def _add_triangle_plot_to_axes(self, posterior_samples, theta_0, plot_bounds, axes):

        assert axes.shape == (self.num_dim, self.num_dim)
        
        if plot_bounds is None:
            plot_bounds = [None for _ in range(self.num_dim)]
        else:
            plot_bounds = plot_bounds.T

        for i in range(self.num_dim):
            for j in range(self.num_dim):
                if i < j:
                    h, xedges, yedges , _ =axes[i,j].hist2d(posterior_samples[:,j], posterior_samples[:,i], range= [plot_bounds[j], plot_bounds[i]], bins=50, density=True)
                    axes[i,j].vlines([theta_0[j]], ymin=np.min(yedges), ymax = np.max(yedges), color="red", linestyle='--', label="theta")
                    axes[i,j].hlines([theta_0[i]], xmin=np.min(xedges), xmax = np.max(xedges), color="red", linestyle='--', label="theta")
                elif i == j:
                    n, _ ,_ = axes[i,j].hist(posterior_samples[:,i], range=plot_bounds[i], bins=50, density=True)
                    axes[i,j].vlines([theta_0[i]], ymin=np.min(n), ymax = np.max(n), color="red", linestyle='--', label="theta")
                    axes[i,j].set_xlabel(f"{self.parameters_info[i].name} ({self.parameters_info[i].unit})")
                    axes[i,j].tick_params(left = False, labelleft = False)
                    if i == 0:
                        axes[i,j].legend()
                else:
                    axes[i,j].set_visible(False)

    def plot_beachball_samples(self, inversion_data, plot_path : Path = None):
        theta0, samples, data_scaler, _ = inversion_data
        if data_scaler is None:
            data_scaler = self.data_scaler
        plotting_units_samples = data_scaler.inverse_transform(samples)
        if theta0 is not None:
            plotting_units_theta_0 = data_scaler.inverse_transform(theta0.reshape(1, -1)).flatten()
        else:
            plotting_units_theta_0 = None

        np.random.shuffle(plotting_units_samples)
        if plot_path is not None:
            filename = plot_path.stem
            figsave_1 = plot_path.with_name(f"{filename}_samples.png")
            figsave_2 = plot_path.with_name(f"{filename}_fuzzy.png")
        else:
            figsave_1 = None
            figsave_2 = None
        self.plot_seperate_beachballs(plotting_units_samples, plotting_units_theta_0, figsave=figsave_1)
        self.plot_fuzzy_beachball_samples(plotting_units_samples, plotting_units_theta_0, figsave=figsave_2)

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
    
    def plot_fuzzy_beachball_samples(self, plotting_units_samples, plotting_units_theta_0, sample_color='cornflowerblue', figsave = None):

        pyrocko_mts = []
        np.random.shuffle(plotting_units_samples)
        for mt in plotting_units_samples[:1000]:
            pyrocko_mts.append(mtm.MomentTensor(m_up_south_east=create_matrix(self.convert_mt_convention(mt))))

        plot_kwargs = {
            'beachball_type': 'full',
            'size': 8,
            'position': (5, 5),
            'color_t':sample_color,
            'color_p':'white',
            'edgecolor':'black',
            'best_color':'black',
            'linewidth':5,
            'alpha':1,
            }

        fig = plt.figure(figsize=(10., 10.))
        axes = fig.add_subplot(1, 1, 1)
        #remove ticks and labels 
        axes.set_xticks([])
        axes.set_yticks([])
        # remove figure box
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['bottom'].set_visible(False)
        axes.spines['left'].set_visible(False)
        if plotting_units_theta_0 is not None:
            rocko_beachball.plot_fuzzy_beachball_mpl_pixmap(pyrocko_mts, axes, mtm.MomentTensor(m_up_south_east=create_matrix(self.convert_mt_convention(plotting_units_theta_0))), **plot_kwargs)
        else:
            rocko_beachball.plot_fuzzy_beachball_mpl_pixmap(pyrocko_mts, axes, **plot_kwargs)

        current_xlim = axes.get_xlim()
        current_ylim = axes.get_ylim()

        # Modify the limits as needed
        new_xlim = (current_xlim[0]- 0.1, current_xlim[1] + 0.1)  # Replace with your desired values
        new_ylim = (current_ylim[0] + 0.1, current_ylim[1] - 0.1)  # Replace with your desired values

        # Set the new x-axis and y-axis limits
        axes.set_xlim(new_xlim)
        axes.set_ylim(new_ylim)
        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave)
        plt.close()
    
    @staticmethod
    def convert_mt_convention(mt_rr_phi_theta):
        """(mnn, mee, mdd, mne, mnd, med)"""

        return [mt_rr_phi_theta[0], mt_rr_phi_theta[1], mt_rr_phi_theta[2], mt_rr_phi_theta[3], -mt_rr_phi_theta[4], -mt_rr_phi_theta[5]]


    @staticmethod
    def add_beachball_plot(ax, name, moment_tensor_sol, M0_epsilon, col = 'b', add_text = True):
        if add_text:
            extra_text = f"\n $M_W=${M0_epsilon[0]:.3f},\n$\epsilon= {M0_epsilon[1]:.2f}$"
        else:
            extra_text = ""
        ax.axis('off')
        ax.set_title(f"{name}{extra_text}")
        ax.add_collection(beach(moment_tensor_sol, width=1.5, facecolor=col))
        ax.set_aspect("equal")
        ax.set_xlim((-0.8, 0.8))  
        ax.set_ylim((-0.8, 0.8))
        