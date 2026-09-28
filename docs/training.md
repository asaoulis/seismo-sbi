# Neural compression and NPE training

The [inference pipeline](pipeline.md) compresses the data with a linearised score compressor and
trains a density estimator for one event at a time. This page covers the other route: a neural
network that compresses the seismograms of every station to a short summary, trained together
with a conditional normalising flow over the source parameters. The result is an amortised
neural posterior estimator (NPE). Once trained on simulations drawn from the prior, it returns the
posterior of any new event recorded by the same network in a fraction of a second, with no
further simulation. This is what builds whole moment-tensor catalogues.

One YAML file drives training: the same blocks as an inversion (see
[the configuration guide](configuration.md)) plus the `ml_*` blocks below.
`examples/configs/npe_example.yaml` is a small complete example, and the
[`03_npe_training_and_evaluation`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/03_npe_training_and_evaluation.ipynb)
notebook runs it step by step.

## Running a training

```bash
python scripts/train_NPE.py --config examples/configs/npe_example.yaml --run-name my_model
```

| flag | effect |
|---|---|
| `--stage generate` | simulate the training set and stop |
| `--stage train` | simulate if needed, then train (the default) |
| `--stage meta` | write the run's `model_meta.json` without training |
| `--epochs`, `--devices`, `--architecture`, `--train-batch-size`, `--num-simulations` | override the configuration for this run |

With `--devices` above one, training runs one process per GPU. On a cluster the two stages
usually run as separate jobs: `generate` on CPU nodes, `train` on GPU nodes over the same
simulations.

Outputs, under the configuration's `output_directory`:

| path | contents |
|---|---|
| `sims/<run_name>/<job_name>/` | the training simulations, one HDF5 file each |
| `<run_name>/<job_name>/<--run-name>/` | the checkpoints, `model_meta.json` (the architecture and parameter scaling needed to rebuild the model) and `metrics.csv` |

## The steps of a training

In Python, `train_NPE.py` reads:

```python
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.datasets.training_data import build_pipeline, generate_training_dataset, prepare_training_data
from seismo_sbi.sbi.compression.ML.train import CompressionTrainer, attach_loggers
from seismo_sbi.sbi.scalers import scaler_provenance

config = SBI_Configuration.from_file("examples/configs/npe_example.yaml")
training = config.training

pipeline = build_pipeline(config, "examples/configs/npe_example.yaml")
simulation_paths = generate_training_dataset(pipeline, config, training.skip_compression_stencil)

data = prepare_training_data(pipeline, config, simulation_paths, training)
trainer = CompressionTrainer.from_configuration(
    training, data.components, data.station_locations, data.trace_length,
    scaler_provenance(data.data_scaler))
trainer.train("my_model", epochs=training.epochs, output_path=pipeline.models_output_path,
              dataloader_args=training.dataloader_args(pipeline, data),
              logger=attach_loggers(training.logging, pipeline.models_output_path / "my_model"),
              devices=training.devices)
```

1. **Forward model and prior.** `build_pipeline` builds the forward model, the stations and the
   parameter samplers the configuration describes.
2. **Training set.** `generate_training_dataset` simulates `jobs.simulations.random_events`
   sources drawn from the prior, one HDF5 file each. Nuisance parameters staged `simulation`
   (for example a Green's-function ensemble member per simulation, or the source time function's
   duration) are drawn here and baked into the seismograms.
3. **Noise, augmentation and scaling.** `prepare_training_data` sets up:
   - the training noise model (`inference.sbi.noise_model`, typically recorded noise windows);
   - the augmentations: nuisance effects staged `training_augmentation` (amplitude errors, time
     shifts, station dropout, scattering coda, ...), applied afresh every time a simulation is
     drawn;
   - the scaler that maps the source parameters to the unit box the flow works in.
4. **Model.** `CompressionTrainer` builds the embedding network and the flow (see below).
5. **Training.** `train` maximises the flow's log-probability of the true source given the
   noisy, augmented seismograms. The last `1 − ml_batch.train_fraction` of the simulations are
   held out for validation. Checkpoints are kept by validation loss.

The [`nuisance_parameters_demo`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/nuisance_parameters_demo.ipynb)
and [`nuisance_augmentation_demo`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/nuisance_augmentation_demo.ipynb)
notebooks show the two nuisance stages.

## The model

The input is every trace of every station, `(n_stations, n_components, n_samples)`.

1. **Station encoder.** Each station's traces are encoded separately into a short sequence of
   feature vectors. `ml_architecture` chooses the encoder: `cnn` (convolutional), `tcn`
   (temporal convolutional) or `pno` (a Fourier neural-operator encoder). `ml_encoder` sets its size.
2. **Across stations.** A transformer combines the stations, each tagged with its position
   (`ml_positional_encoding`) and, optionally, conditioned on the source location
   (`ml_conditioning`). The stations form a set, so a model trained with
   `ml_variable_stations` also accepts events recorded by a subset of the network.
3. **Summary.** The station features are pooled (`ml_pooling`) into the summary vector.
4. **Flow.** A conditional masked autoregressive flow (`ml_flow`) models the posterior of the
   scaled source parameters given the summary.

The source parameters are scaled to the unit box of their bounds. `ml_scaler` can instead
scale the moment tensor as its log scalar moment and a unit tensor, which suits catalogues that
span several magnitude units.

## The `ml_*` blocks

| block | sets |
|---|---|
| `ml_architecture`, `ml_encoder` | the station encoder and its size |
| `ml_conditioning` | source-location conditioning of the encoder |
| `ml_variable_stations` | training on random subsets of the station set |
| `ml_positional_encoding`, `ml_amplitude_embedding` | the features each station token carries |
| `ml_pooling`, `ml_summary_bottleneck` | how the station features become the summary |
| `ml_flow` | the normalising flow |
| `ml_optimizer` | learning rate, weight decay and schedule |
| `ml_batch` | batch sizes, dataloader workers and the train/validation split |
| `ml_cache` | holding the simulations and noise in memory during training |
| `ml_scaler` | the parameter scaling |
| `ml_mmd` | an optional extra loss that pulls the summaries of simulated and real data together |
| `ml_warm_start` | starting from an earlier run's weights (`from_run_name`) |
| `ml_perf` | mixed precision and other training-speed options |
| `ml_logging` | `metrics.csv`, Weights & Biases, or both |

An `ml_*` block the training configuration does not know is an error.

## Validation and calibration

`run_validation` (in `seismo_sbi.evaluation.validation`) samples the trained posterior for
held-out simulations, drawn with the same noise and augmentation the model trained on.
`write_validation_outputs` turns the result into:

- recovery plots of the posterior against the true source, per parameter;
- a TARP coverage test (Lemos et al. 2023): for a calibrated posterior, the expected coverage of
  its credible regions equals their credibility;
- a metrics JSON with the bias, width and coverage per parameter, including the derived
  source-type (γ, δ), Mw, strike, dip and rake.

```python
from copy import deepcopy
from seismo_sbi.evaluation.validation import run_validation, write_validation_outputs

original_parameters = deepcopy(pipeline.parameters)   # taken right after generate_training_dataset
posterior = trainer.build_posterior()
validation = run_validation(pipeline, original_parameters, posterior, data.data_scaler)
summary = write_validation_outputs(validation, "validation/", original_parameters,
                                   data.data_scaler, num_samples=1000)
```

The notebook's last two sections run this and show the figures.

## Using a trained model on real data

`seismo_sbi.evaluation.inference` rebuilds a trained model and loads real events:

- `build_eval_pipeline(config_path)` builds the pipeline from the training configuration;
- `build_ml_posterior(resolve_ckpt_dir(run_directory), pipeline)` rebuilds the embedding network
  and flow from a run directory's `model_meta.json` and best checkpoint, as an `sbi` posterior;
- `load_real_observation(config, pipeline, event_name)` loads an event listed under
  `jobs.real_events` as an `(n_stations, n_components, n_samples)` array.

The parameter scaling at inference must match the one the model was trained with:
`build_flexible_scaler(parameters, raw_config, model_meta=meta)` builds it from the
configuration and the scalar-moment convention recorded in `model_meta.json` (checkpoints that
record none were trained with the six-component moment), and `check_scaler_provenance`
compares it with the record.
