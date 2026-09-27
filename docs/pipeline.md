# Inference pipeline

A run inverts one earthquake for its source parameters with two methods side by side:
simulation-based inference (SBI) and a Gaussian-likelihood MCMC. Both work from the same data,
forward model and score compression, so their posteriors can be compared directly. The run can
also invert synthetic test events with known sources, to check the posteriors against the truth.

One YAML file drives the whole run (see [the configuration guide](configuration.md));
`examples/configs/LV2.yaml` is a complete example, and the
[`theory_errors_LV2`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/theory_errors_LV2.ipynb)
notebook runs it step by step.

## Running an inversion

```bash
python scripts/event_inversion.py --config examples/configs/LV2.yaml
```

Outputs, under the configuration's `output_directory`:

| path | contents |
|---|---|
| `plots/<run_name>/<job_name>/` | posterior, misfit and comparison figures, and a copy of the configuration |
| `sims/<run_name>/<job_name>/` | the test-event and training simulations |
| `jobs/<run_name>/<job_name>/inversion_results.pkl` | `(job_data, job_results, inversion_results)` for every event and method |

## The steps of a run

`SingleEventPipeline` (in `seismo_sbi.sbi.pipeline`) holds the state the steps share. In
Python, a run reads:

```python
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.pipeline import SingleEventPipeline
from seismo_sbi.sbi import job_runners

config = SBI_Configuration()
config.parse_config_file("examples/configs/LV2.yaml")

pipeline = SingleEventPipeline(config.pipeline_parameters, "examples/configs/LV2.yaml")
pipeline.compression_methods = config.compression_methods
pipeline.seed = config.sbi_seed
pipeline.load_seismo_parameters(config.sim_parameters, config.model_parameters, config.dataset_parameters)

test_jobs_paths = pipeline.simulate_test_jobs(config.dataset_parameters, config.test_job_simulations)
pipeline.compute_data_vector_properties(test_jobs_paths, config.real_event_jobs)
compression_data, extra_gradients = pipeline.compute_required_compression_data(
    config.compression_methods, config.model_parameters)
pipeline.load_compressors(config.compression_methods, compression_data,
                          extra_gradients=extra_gradients, freeze=True)
pipeline.load_test_noises(config.sbi_noise_model, config.test_noise_models)

job_data = pipeline.create_job_data(test_jobs_paths, config.real_event_jobs)
results = pipeline.run_compressions_and_inversions(
    job_data, config.sbi_method, config.likelihood_config, config.dataset_parameters)
job_results, inversion_results = job_runners.run_all_inversions_before_plotting(pipeline, results)
```

1. **Parameters and forward model.** `load_seismo_parameters` sets the source parameters to
   invert for and their bounds, the nuisance parameters, the forward model and the stations.
2. **Test events.** `simulate_test_jobs` simulates the synthetic events listed under
   `jobs.simulations`. Real events come from the HDF5 files listed under `jobs.real_events`.
3. **Compression data.** `compute_required_compression_data` computes the derivatives of the
   synthetics with respect to each source parameter about the fiducial source. For a
   theory-error compressor it also simulates the Earth-model ensemble, from which the
   theory-error covariance is estimated.
4. **Compressors.** `load_compressors` builds one score compressor per entry of the
   `compression` block (see below).
5. **Noise.** `load_test_noises` builds the noise added to the synthetic test events
   (`jobs.noise_models`) and the noise added to every training simulation
   (`inference.sbi.noise_model`).
6. **Jobs.** `create_job_data` makes one job per event and test noise: the data vector, the
   true source when it is known, and the event's noise covariance.
7. **Inversions.** `run_compressions_and_inversions` runs, for each job and each compressor,
   the SBI inversion and then, if `inference.likelihood.run` is set, the Gaussian-likelihood
   inversion. It yields one result at a time.

`MultiEventPipeline` and `VaryDatasetSizeEventPipeline` (in `seismo_sbi.sbi.pipeline_variants`)
run the same steps over several events, or over training sets of increasing size. Choose one with
`inference.sbi.pipeline`: `single_event`, `multi_event` or `vary_dataset_size`.

## Score compression

A compressor maps the data vector (every trace of every station, concatenated) to one number per
source parameter: the maximum-likelihood estimate under a Gaussian likelihood, linearised about a
fiducial source. It needs the derivatives from step 3 and a data covariance. Each entry of the
`compression` block is one compressor, and each runs its own inversions:

```yaml
compression:
  theory_optimal_score:
    noise_level: 1.e-8
    diag_regularisation_magnitude: 0.005
    data_covariance: kolb
```

| compressor | covariance |
|---|---|
| `optimal_score` | a data-noise covariance, chosen from the table below |
| `theory_optimal_score` | a data-noise covariance (`data_covariance`, with `noise_level`) plus the theory-error covariance from the Earth-model ensemble, per trace; `diag_regularisation_magnitude` adds to its diagonal |
| `second_order_score` | a scalar noise level; adds the second-order derivatives |
| `multi_optimal_score` | a scalar noise level; compresses about several fiducial points |
| `ml_compressor` | none: a trained neural compressor (see `scripts/train_NPE.py`) |

| data covariance | structure |
|---|---|
| `noise_level` | white noise of one variance |
| `empirical_diagonal` | one variance per trace, measured from recorded noise |
| `empirical_block` | a Toeplitz block per trace, from the measured noise autocovariance |
| `filtered_block` | a Toeplitz block per trace: white noise through the processing filter |
| `kolb` | a Toeplitz block per trace with correlation e^(−λ\|Δt\|) cos(λω₀\|Δt\|) |

The [`02_noise_covariances_and_likelihood`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/02_noise_covariances_and_likelihood.ipynb)
notebook builds every covariance from synthetic noise and checks the compression against the
analytical posterior.

## The SBI inversion

For each job and compressor:

1. **Maximum-likelihood estimate.** Iterative least squares from the fiducial source finds the
   MLE, and the compressor is re-centred on it. `simulations.iterative_least_squares` sets the
   iterations; `mcmc_chain_for_mle` adds rounds of MCMC refinement.
2. **Training set.** `simulations.num_simulations` sources are drawn uniformly from the prior
   box: the parameter bounds, narrowed to ± `simulations.use_fisher_to_constrain_bounds` Fisher
   standard deviations about the MLE (default 5; 0 keeps the bounds). Each is simulated, given a
   noise realisation from the training noise model, and compressed.
3. **Density estimation.** A neural density estimator is trained on the (source, compressed
   data) pairs with [`sbi`](https://github.com/sbi-dev/sbi): `inference.sbi.method: posterior`
   trains a posterior estimator (SNPE-C with a masked autoregressive flow), and
   `likelihood` trains a likelihood estimator (SNLE).
4. **Posterior.** The posterior is sampled at the compressed observed data (10 000 samples).

Parameters are scaled to the unit box of their bounds for training and sampling, and returned
in physical units. `inference.sbi.seed` makes the SBI inversion reproducible.

## The Gaussian-likelihood inversion

The Gaussian-likelihood inversion samples

log p(θ | d) = −½ (s(θ) − d)ᵀ C⁻¹ (s(θ) − d) + log p(θ),

where s(θ) are the synthetics from the fiducial Earth model, d the data, C the covariance and
p(θ) uniform over the parameter bounds. It uses the forward model at every step, so it is the
reference the SBI posterior is compared with. Sampling is with [emcee](https://emcee.readthedocs.io).
The `inference.likelihood` block sets it:

```yaml
inference:
  likelihood:
    run: True
    covariance: empirical     # the compressor's covariance
    ensemble: False           # True: emcee's affine-invariant ensemble; False: independent Gaussian-move chains
    num_samples: 20000
    walker_burn_in: 500
    move_size: [0.0001, 0.0001]
```

`num_samples` is split between the chains (one per process, `num_processes`, by default the
run's `num_jobs`). `move_size` is the proposal step of the Gaussian-move chains in the unit-box
parameter coordinates: one value, or `[burn-in step, sampling step]`. With `covariance: empirical`, each compressor's inversion uses that
compressor's covariance, so each covariance model gets its own Gaussian-likelihood posterior.

### Standalone use

The likelihood and sampler work without the pipeline. `GaussianLikelihoodEvaluator` takes the
data, any function from source parameters to a data vector, a scaler to the unit box, and the
covariance's loss; `generate_samples` runs the chains:

```python
from seismo_sbi.sbi.likelihood import GaussianLikelihoodEvaluator, generate_samples
from seismo_sbi.sbi.scalers import ZeroOneScaler

scaler = ZeroOneScaler((lower_bounds, upper_bounds))
evaluator = GaussianLikelihoodEvaluator(
    observed_data, forward_model, scaler,
    covariance.create_loss_callable(covariance.inverse_metadata, covariance.data_vector_length))
samples = generate_samples(evaluator.log_probability, False, num_parameters,
                           nsamples_per_walker=5000, nwalkers=4, burn_in=1000, num_processes=4)
posterior_samples = scaler.inverse_transform(samples)
```

The `02_noise_covariances_and_likelihood` notebook does this for a linear forward model and
checks each sampled posterior against the analytical one.
