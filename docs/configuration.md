# Configuration

A pipeline is driven by one YAML file, parsed once by
`seismo_sbi.sbi.configuration.SBI_Configuration.from_file`. Training options are parsed from
the same file into `seismo_sbi.sbi.training_configuration.TrainingConfiguration`. Working
examples: `examples/configs/LV2.yaml` (Gaussian likelihood and SBI) and
`examples/configs/npe_example.yaml` (NPE training).

## Blocks

| block | parsed into | holds |
|---|---|---|
| top-level scalars | `PipelineParameters` | `run_name`, `output_directory`, `job_name`, `generate_dataset`, `num_jobs` |
| `seismic_context` | `SimulationParameters` (`sim_parameters`) | forward model, stations, components, seismogram length and rate, processing |
| `parameters` | `ModelParameters` (`model_parameters`) | `inference` and `nuisance` parameters: fiducial values, stencil deltas, bounds, and each nuisance's `stage` |
| `simulations` | `DatasetGenerationParameters` (`dataset_parameters`) | `num_simulations`, per-parameter `sampling_method`, iterative least squares |
| `compression` | `compression_methods` | score compressors and their options, or `{}` for none |
| `inference` | `sbi_method`, `likelihood_config` | the SBI method, pipeline type, training noise model, Gaussian-likelihood options |
| `jobs` | `test_job_simulations`, `real_event_jobs` | synthetic test events, noise models to test against, real events, plots |
| `ml_*` | `TrainingConfiguration` (`training`) | NPE architecture, encoder, conditioning, flow, optimiser, batches, caches, logging, scaler |

The `seismic_context`, `parameters`, `simulations`, `compression`, `inference` and `jobs` blocks
are required. An `ml_*` block that the training configuration does not know is an error.

## `seismic_context.processing`

How the synthetics are filtered so that they match the observed data:

```yaml
seismic_context:
  sampling_rate: 1.0            # Hz, the rate of the data vector
  seismogram_duration: 200      # s
  processing:
    filter_sampling_rate: 20.0  # Hz, the rate the observed data are filtered at
    filter:                     # keyword arguments of obspy's Stream.filter
      type: 'bandpass'
      freqmin: 0.02
      freqmax: 0.08
      corners: 4
      zerophase: False
    sampling_rate: 1.0          # Hz, the rate the filtered synthetics are resampled to
```

Three rates sit next to each other:

- `seismic_context.sampling_rate`, with `seismogram_duration`, sets the length of the data
  vector.
- `processing.sampling_rate` is the rate the Instaseis backend resamples the filtered synthetics
  to, over `seismogram_duration`.
- `processing.filter_sampling_rate` is the rate the filter is designed at. The Instaseis backend
  evaluates that filter's response at the synthetics' own frequencies and applies it in the
  frequency domain, so they see the same filter response as the observed data without being
  resampled to that rate. Set it to the raw rate of the recordings, the rate the
  data preparation filtered them at (`build_catalogue.py` filters each channel at its raw rate
  before resampling). For synthetic-only work, any rate comfortably above twice `freqmax`
  will do.

`filter_sampling_rate` is required in every configuration, whatever the backend. A file
without it fails to parse with `InvalidConfiguration`. The test suite checks that every committed
configuration sets it above twice `filter.freqmax`. Only the Instaseis backend reads it at
present; the CPS backend filters at the sampling rate of its Green's functions.

The filtered synthetics start 60 s (`SYNTHETICS_PRE_EVENT_PAD_S`) before the origin exactly,
on the sample grid through that instant, whatever the database's own sample interval.
