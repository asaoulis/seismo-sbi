from typing import List

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import Affine2D

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

from .parameters import ParameterInformation
import torch
from .patched_chainconsumer import CustomChainConsumer as ChainConsumer
from obspy.imaging.beachball import beach
from obspy.imaging import beachball
from pyrocko.plot import beachball as rocko_beachball
import pyrocko.moment_tensor as mtm
from ..instaseis_simulator.dataset_generator import tqdm_joblib
from .rocko_beachball_patch import plot_beachball_on_axes
from tqdm import tqdm
from contextlib import contextmanager
import logging
from seismo_sbi.plotting.MTfit import _LunePlot
from pyrocko import moment_tensor as pmt
# New: reusable lune plotting utilities
from seismo_sbi.plotting.lune import (
    mts6_to_gamma_delta,
    plot_lune_frame,
    kde_on_grid,
    kde_hpd_contour_levels,
)

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

def make_beachball_collection(mt, facecolor, edgecolor, alpha=1.0, linewidth=1.3):
    # Get raw, unit beachball polygons
    data = rocko_beachball.mt2beachball(
        mt,
        beachball_type='full',
        position=(0., 0.),
        size=.06
    )

    patches = []
    for (path, fc, ec, lw) in data:
        patches.append(
            Polygon(
                xy=path,
                facecolor=facecolor if fc != 'none' else 'none',
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha
            )
        )

    return PatchCollection(patches, match_original=True)



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

# New: compute delta (Tape & Tape lune coordinate) from full tensor eigenvalues
# delta is the angle from the deviatoric plane to the lune point (-90 <= delta <= 90)
# Following TT2012 Eq. 21a and the reference lam2lune.m

def get_delta(moment_tensor_sol: np.ndarray) -> float:
    """
    Compute Tape & Tape lune delta (in degrees) for a 6-component moment tensor.
    Input ordering matches create_matrix: [Mxx, Myy, Mzz, Mxy, Mxz, Myz] in up-south-east.
    """
    M = create_matrix(moment_tensor_sol)
    # For symmetric tensors, eigvalsh is faster and yields ordered real vals (ascending)
    lam = np.linalg.eigvalsh(M)[::-1]  # descending: lam1 >= lam2 >= lam3
    rho = float(np.sqrt(np.sum(lam**2)))
    if rho == 0.0:
        return 0.0
    trM = float(np.sum(lam))
    # numerical safety: if trace(M) == 0 => delta = 0
    if np.isclose(trM, 0.0, atol=1e-12, rtol=0.0):
        return 0.0
    bdot = trM / (np.sqrt(3.0) * rho)
    bdot = float(np.clip(bdot, -1.0, 1.0))
    delta = 90.0 - np.degrees(np.arccos(bdot))
    return float(delta)

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

                        
    parameters_info = [
                        ParameterInformation("$\delta$", "°"),
                        ParameterInformation("$\gamma$", '°'),
                        ParameterInformation("$M_w$", ""),
                        ParameterInformation(r"$\textrm{strike}$", '°'),
                        ParameterInformation(r"$\textrm{dip}$", '°'),
                        ParameterInformation(r"$\textrm{rake}$", '°'),
                        ]
    
    def __init__(self, data_scaler, parameters):
        # to convert samples from nde training scale
        # back to natural units
        self.data_scaler = data_scaler
        self.parameters = parameters
        # no need for an extra scaling step
        dummy_scaler = DummyDataScaler(5)
        self.chain_plotter = PosteriorPlotter(dummy_scaler, self.parameters_info)

    def convert_samples(self, samples, theta0, custom_select):
        
        converted = []
        for sample in samples:
            inputs = self.parameters.vector_to_simulation_inputs(sample, only_theta_fiducial=True)
            mt = inputs["moment_tensor"]
            # Use lune utilities to compute gamma and delta (in degrees)
            g, d = mts6_to_gamma_delta(np.asarray(mt).reshape(1, -1))
            # Keep Mw from scalar moment
            MW, _ = get_MW_and_epsilon(mt)
            # Choose nodal plane
            nodal_plane_pair = get_nodal_planes(mt)
            nodal_plane = nodal_plane_pair[0] if not custom_select else custom_select(nodal_plane_pair)
            converted.append(np.array([float(g[0]),float(d[0]), MW, nodal_plane[0], nodal_plane[1], nodal_plane[2]]))
        if theta0 is not None:
            theta_inputs = self.parameters.vector_to_simulation_inputs(theta0, only_theta_fiducial=True)
            theta_mt = theta_inputs["moment_tensor"]
            tg, td = mts6_to_gamma_delta(np.asarray(theta_mt).reshape(1, -1))
            theta_MW, _ = get_MW_and_epsilon(theta_mt)
            nodal_plane_pair = get_nodal_planes(theta_mt)
            nodal_plane = nodal_plane_pair[0] if not custom_select else custom_select(nodal_plane_pair)
            theta0_converted = np.array([float(tg[0]), float(td[0]), theta_MW, nodal_plane[0], nodal_plane[1], nodal_plane[2],])
        else:
            theta0_converted = None
        return np.array(converted), theta0_converted



    def plot_chain_consumer(self, samples_theta0_dict, custom_processing = None, *args, **kwargs):
        converted_chain_dict = self.convert_chain_dict(samples_theta0_dict, custom_processing)

        with warning_logging_disabled():
            self.chain_plotter.plot_chain_consumer(converted_chain_dict,  *args, **kwargs)

    def convert_chain_dict(self, samples_theta0_dict, custom_processing):
        converted_chain_dict = {}
        for name, (theta0, samples, data_scaler, _) in samples_theta0_dict.items():
            if data_scaler is None:
                data_scaler = self.data_scaler
            # samples = data_scaler.inverse_transform(samples)
            if theta0 is not None:
                pass
                # theta0 = self.data_scaler.inverse_transform(theta0.reshape(1, -1)).flatten()
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
            c_plot.add_chain(samples, parameters=parameters_label, color=colors[i], name=name, shade=shade, linewidth=2.5)
            i+=1
        c_plot.configure(kde=[kde for _ in range(len(inversion_data))], shade_alpha=0.7, max_ticks=3, diagonal_tick_labels=False, inverse=inverse, tick_font_size=tick_font_size, label_font_size=40, summary=False, usetex=True, bar_shade=True)
        c_plot.configure_truth(lw=2)
        scale = 2.8*self.num_dim

        fig = c_plot.plotter.plot(figsize=(scale,scale), truth=truth, legend=False, extents=extents)
        fig.align_labels() 

        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight")
        plt.close()
    
    def plot_lunes(self, inversion_data, num_samples=250, plot_beachballs=True, figsave=None):

        # New implementation: project ensembles onto the standard Tape & Tape lune (Hammer) and scatter
        fig, ax = plt.subplots(figsize=(14, 14))
        bm = plot_lune_frame(ax)

        colors = ['cornflowerblue', 'red', 'purple', 'green', 'brown']
        true_theta0 = None



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
                # if plot beachballs for true, plot 3 beachballs and truth
                true_mt = true_theta0
                percentile_mts = []
                for q in [5, 50, 95]:
                    d_q = np.percentile(delta, q)
                    # find closest sample to this delta
                    idx = np.argmin(np.abs(delta - d_q))
                    percentile_mts.append(samples_MT[idx])

                # add true beachball in gold
                for idx, mt in enumerate([true_mt] + percentile_mts):
                    if mt is None:
                        continue
                    facecolor = 'black' if idx == 0 else colors[i % len(colors)]
                    mt = np.array(self.convert_mt_convention(mt))
                    tg, td = mts6_to_gamma_delta(mt.reshape(1, -1))
                    tx, ty = bm(tg, td)
                    mt = mtm.MomentTensor(m_up_south_east=create_matrix(mt))
                    plot_beachball_on_axes(ax, mt, tx[0], ty[0], diameter=0.08, color_t=facecolor, edgecolor='black', zorder=10, linewidth=0.5)
                    ax.scatter(tx, ty, color=facecolor, alpha=1.0, marker='o', s=20, zorder=11)



        if figsave is None:
            plt.show()
        else:
            fig.savefig(figsave, dpi=200, transparent=True, bbox_inches="tight")
        plt.close()


    def plot_lunes_kde(self, inversion_data, num_samples=2500, plot_beachballs=True, plot_inset=False, figsave=None, ax=None, show=True):
        """Plot 68%/95% HPD KDE contours for each ensemble on the projected lune, with a zoomed inset around truth ±8°."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 14))
        else:
            fig = ax.get_figure()
        
        bm = plot_lune_frame(ax)

        colors = ['cornflowerblue', 'red', 'purple', 'green', 'brown']
        qs = [[5,50,95], [50], [5,50,95]]
        gx = np.linspace(-30, 30, 200)
        gy = np.linspace(-90, 90, 300)
        GX, GY = np.meshgrid(gx, gy)
        XX, YY = bm(GX, GY)

        # Cache per-ensemble gamma/delta for reuse in inset and store truth from first ensemble
        gd_list = []
        true_theta0 = None

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
                # if plot beachballs for true, plot 3 beachballs and truth
            true_mt = true_theta0
            percentile_mts = []
            for q in qs[i]:
                d_q = np.percentile(d, q)
                # find closest sample to this delta
                idx = np.argmin(np.abs(d - d_q))
                percentile_mts.append(samples_MT[idx])

            # add true beachball in gold
            fig.canvas.draw()

            for idx, mt in enumerate([true_mt] + percentile_mts):
                if mt is None:
                    continue

                is_true_mt = (idx == 0)

                # If beachballs are OFF, only plot the true MT (scatter only)
                if not plot_beachballs and not is_true_mt:
                    continue

                mt = np.array(self.convert_mt_convention(mt))

                facecolor = 'peru' if is_true_mt else colors[i % len(colors)]
                marker = 'd' if is_true_mt else 'o'

                tg, td = mts6_to_gamma_delta(mt.reshape(1, -1))
                tx, ty = bm(tg, td)
                mt = mtm.MomentTensor(m_up_south_east=create_matrix(mt))

                # Plot beachball only if enabled
                if plot_beachballs:
                    plot_beachball_on_axes(
                        ax,
                        mt,
                        tx[0],
                        ty[0],
                        diameter=0.08,
                        color_t=facecolor,
                        edgecolor='black',
                        zorder=10,
                        linewidth=1
                    )

                ax.scatter(
                    tx,
                    ty,
                    color=facecolor,
                    alpha=1.0,
                    marker=marker,
                    s=320 if is_true_mt else 40,
                    zorder=11
                )
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
        # raw_units_samples = data_scaler.inverse_transform(samples)
        if theta0 is not None:
            # raw_units_theta_0 = data_scaler.inverse_transform(theta0.reshape(1, -1))
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
        # plotting_units_samples = data_scaler.inverse_transform(samples)
        # if theta0 is not None:
        #     plotting_units_theta_0 = data_scaler.inverse_transform(theta0.reshape(1, -1)).flatten()
        # else:
        #     plotting_units_theta_0 = None

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


    def add_beachball_plot(self, ax, name, moment_tensor_sol, M0_epsilon, col = 'b', add_text = True):
        mt = mtm.MomentTensor(m_up_south_east=create_matrix(self.convert_mt_convention(moment_tensor_sol)))
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


# -----------------------------
# Standalone plotting: histories (trajectories) on the lune
# -----------------------------

def plot_lune_histories(histories, figsave=None, linewidth=2.0, mark_endpoints=True):
    """
    Plot trajectories of MT histories on the Tape & Tape lune (Hammer projection).

    histories: dict mapping name -> sequence of MTs in 6-component form [Mxx, Myy, Mzz, Mxy, Mxz, Myz].
               Each value can be an array of shape (T, 6) or an iterable of length T with 6-vectors.
    figsave: optional path to save the figure; if None, shows the plot.
    linewidth: line width for the trajectory.
    mark_endpoints: if True, mark start (circle) and end (cross) points of each trajectory.
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