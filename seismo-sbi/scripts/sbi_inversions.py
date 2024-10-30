import sys
import os
from pathlib import Path
import pickle
import joblib
import torch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname('.'), '..')))

import numpy as np

from src.instaseis_simulator.receivers import Receivers

from src.sbi.compression.gaussian import GaussianCompressor, MachineLearningCompressor
from src.sbi.compression.ML.seismogram_transformer import LightningModel
from src.instaseis_simulator.dataloader import SimulationDataLoader

from src.sbi.pipeline import SBIPipeline, SBIPipelinePlotter
from src.plotting.distributions import PosteriorPlotter
from src.plotting.parameters import ParameterInformation, DegreeKMConverter, DegreeType

from sklearn.preprocessing import MinMaxScaler
# Seismo array + source setup

stations_path = './data/UPFLOW_STATIONS_missing'
receivers = Receivers(stations_path)

components = 'Z'
output_folder = Path("./data/sims")

depth = 9
lat = 38.67
long = -28.1
time_shift = 0

theta_fiducial = np.array([lat,  long, depth,time_shift])
stencil_deltas = np.array([0.01, 0.01,   2.5,  0.52    ])

num_dim = theta_fiducial.shape[0]

seismogram_duration = 600

use_local_model = True
if use_local_model:   syngine_address = "/data/shared/prem_i_2s"
else:  syngine_address =  "syngine://prem_i_2s"

# Simulation dataset parameters

num_simulations = 1000

bounds = (np.array([38.66, -28.120, 3, -1]),
          np.array([38.685,-28.076, 20, 1]))

sampling_method = "latin hypercube"

### Start SBI Pipeline

# Generate synthetics dataset

sbi_run_name = "source_time_location_red"

sbi_pipeline = SBIPipeline(sbi_run_name,
                            num_parallel_jobs=8)

sbi_pipeline.load_seismo_parameters(receivers, components, seismogram_duration,
                                    syngine_address=syngine_address)

score_compression_data = sbi_pipeline.compute_compression_data_from_stencil(theta_fiducial, stencil_deltas,  rerun_if_stencil_exists = False)
data_vector_length = score_compression_data.data_parameter_gradients.shape[1]

# sbi_pipeline.generate_dataset(bounds, num_simulations, sampling_method)

# Compressor parameters + apply compression

param_names = ["latitude","longitude", "depth", "time_shift"]

test_size = 10

noise_level = 3e-3

cov_mat = np.diag(noise_level *np.ones((data_vector_length)))

model_config = {"layers": 4,
      "channels": 64,
      "nheads": 8,
      "timeemb": 64,
      "posemb": 64}


train_noise_sampler = lambda : np.random.normal(0, noise_level *np.ones((data_vector_length)))

station_locations = np.array([[rec.latitude, rec.longitude] for rec in receivers.receivers])
scaler = MinMaxScaler()
scaler.fit(station_locations)
scaled_station_locations = torch.Tensor(scaler.transform(station_locations))

ml_model_config = {"num_seismic_components": len(components),
                  "transformer_config" : model_config,
                  "feature_length" : 64,
                  "num_outputs": 4,
                  "noise_model": lambda : torch.Tensor(train_noise_sampler()),
                  "seismogram_locations": scaled_station_locations,
                  "device":'cpu'
            }

def ml_model_preprocessor_4x(d):
    data_shape = d.shape[0]
    num_receivers = len(receivers.receivers)
    num_components = len(components)
    torch_tensor = torch.Tensor(d).reshape((num_receivers, num_components, data_shape//(num_receivers*num_components)))
    downsampled = torch_tensor[:,:,::4]
    return downsampled
def ml_model_preprocessor_2x(d):
    data_shape = d.shape[0]
    num_receivers = len(receivers.receivers)
    num_components = len(components)
    torch_tensor = torch.Tensor(d).reshape((num_receivers, num_components, data_shape//(num_receivers*num_components)))
    downsampled = torch_tensor[:,:,::2]
    return downsampled

simulations_paths = Path("./data/sims/source_time_location_red/")
sim_string = "sim_"
sims_paths = simulations_paths.glob(f"{sim_string}*")
sorted_sims_paths = list(sorted(sims_paths, key = lambda path: int(str(path)[str(path).find(sim_string) + len(sim_string):-3]) ))

param_names = ["latitude","longitude", "depth", "time_shift"]
data_loader = SimulationDataLoader(components, receivers)
source_params = np.zeros((len(sorted_sims_paths),4))
for i, sim_path in enumerate(sorted_sims_paths):  
    inputs = data_loader.load_input_data(sim_path)["source_location"]
    inputs = np.array([inputs[param_name] for param_name in param_names])
    source_params[i] = inputs

from sklearn.utils import shuffle

# source_params = shuffle(source_params, random_state=42)

source_param_scaler = MinMaxScaler()
source_param_scaler.fit(source_params)

compressors = {
    "optimal_score": GaussianCompressor(score_compression_data, cov_mat),
    "ml_compressor": MachineLearningCompressor(LightningModel, "2x_downsample_64ch64emb", ml_model_preprocessor_2x, source_param_scaler, **ml_model_config)
}

real_noise_path = '/home/alex/work/instaseis-sbi/instaseis-sbi/scripts/data/real_noise.npy'

test_noise_configs = {
    "correct_noise": lambda : np.random.normal(0, noise_level *np.ones((data_vector_length))),
}

parameters_info = [
    ParameterInformation("$x$", "km", DegreeKMConverter(bounds[0][0], DegreeType.LATITUDE)),
    ParameterInformation("$y$", "km", DegreeKMConverter(bounds[0][1], DegreeType.LONGITUDE)),
    ParameterInformation("Depth ($z$)", "km"),
    ParameterInformation("Time shift", "s")
]

ml_scaler =  MinMaxScaler()
scaler.fit(station_locations)
scaled_station_locations = torch.Tensor(scaler.transform(station_locations))

samples_dict = {}
for compressor_name, compressor in compressors.items():

    samples_dict[compressor_name] = {}

    raw_compressed_dataset = sbi_pipeline.compress_dataset(compressor, param_names, train_noise_sampler, indices=slice(0,num_simulations-test_size))

    train_data_offset = 0 if compressor_name == "optimal_score" else 190
    train_data, _, data_scaler = sbi_pipeline.prepare_data_for_sbi(raw_compressed_dataset, train_data_offset)
    likelihood_estimator = sbi_pipeline.build_amortised_likelihood_estimator(train_data)

    dist_plotter = PosteriorPlotter(data_scaler, parameters_info)
    for test_noise_name, synthetic_noise_sampler in test_noise_configs.items():
        samples_dict[compressor_name][test_noise_name] = []
        sbi_run_name_eval = "source_time_location_red_eval"

        sbi_run_name_eval = SBIPipeline(sbi_run_name_eval,
                                    num_parallel_jobs=1)

        sbi_run_name_eval.load_seismo_parameters(receivers, components, seismogram_duration,
                                syngine_address=syngine_address,
                                downsampled_length=1171)
        
        test_compressed_dataset = sbi_run_name_eval.compress_dataset(compressor, param_names, synthetic_noise_sampler)
        dist_plotter.plot_compression_errors(test_compressed_dataset)

        test_data, _, _ = sbi_pipeline.prepare_data_for_sbi(test_compressed_dataset, 0, apply_scaling=False)
        test_data = np.hstack([data_scaler.transform(test_data[:, :num_dim]), data_scaler.transform(test_data[:, num_dim:])])

        plotter = SBIPipelinePlotter(sbi_run_name+"_ml", plot_folder_name = f"{compressor_name}/{test_noise_name}")
        plotter.initialise_posterior_plotter(likelihood_estimator, data_scaler, parameters_info)
        
        def generate_samples(test_data_instance):
            theta0, x0 = test_data_instance[:num_dim], test_data_instance[num_dim:]
            samples, _ = plotter.sample_posterior(x0, 10000)
            return theta0, samples

        with joblib.parallel_backend('loky', n_jobs=20):
                samples_dict[compressor_name][test_noise_name] = \
                joblib.Parallel()(
                    joblib.delayed(generate_samples)(test_data_instance) for
                    test_data_instance in test_data[:200]
                )
with open("./data/samples.pkl", 'wb') as fout:
    pickle.dump(samples_dict, fout)