# seismo-sbi

[![Documentation](https://img.shields.io/badge/docs-asaoulis.github.io%2Fseismo--sbi-blue)](https://asaoulis.github.io/seismo-sbi/)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![Version 0.2.0](https://img.shields.io/badge/version-0.2.0-lightgrey)

Full-waveform seismic source inversion using simulation-based inference: improving moment tensor
solutions by accounting for non-Gaussian data and theory errors in full waveform data using
machine learning.

**Documentation:** <https://asaoulis.github.io/seismo-sbi/>. It has the API reference, the
configuration guide, the forward-model guide and the example notebooks rendered with their outputs.

`seismo-sbi` is a Python package for single-event moment-tensor inversion: one earthquake, its
stations and its forward model go in, and a posterior comes out. Its stages can be used on their
own or chained into a full workflow:

- **Forward models:** Instaseis, Computer Programs in Seismology (CPS), Green's-function ensembles
  of perturbed 1-D Earth models, and a registry for plugging in your own.
- **Nuisance effects:** what a real recording does to a synthetic seismogram (amplitude, time shift,
  dropout, scattering coda, anisotropy, dispersion). Each is applied at simulation or training time.
- **Noise and likelihood:** Gaussian-likelihood covariances (diagonal, Toeplitz, theory-block) with
  their estimator and samplers, and real-noise samplers built from recorded noise.
- **Compression and inference:** score compression, neural compressors, and neural posterior
  estimation (NPE) with [`sbi`](https://github.com/sbi-dev/sbi), plus Gaussian-likelihood MCMC for
  comparison.
- **Data preparation and evaluation:** download and preprocessing of real data with ObsPy, data
  quality checks, validation and calibration (TARP), and posterior plots.

## Installation

`seismo-sbi` needs Python 3.11: the current conda-forge `instaseis` requires it. `instaseis` is the
most fragile dependency, so start from a fresh environment and install it first, with the rest of
the scientific, seismology and geospatial stack, from conda-forge. [`environment.yml`](https://github.com/asaoulis/seismo-sbi/blob/main/environment.yml)
does this; `pip` then installs the ML and inference stack:

```
conda env create -f environment.yml      # python 3.11, instaseis, obspy, cartopy, basemap, ...
conda activate seismo-sbi
pip install -e ".[ml,plotting,notebooks]"   # torch, sbi, pytorch_lightning, ChainConsumer, jupyter, ...
```

`pip install -e .` alone installs the forward models, the noise covariances, the Gaussian likelihood
and the score compression (numpy, scipy, emcee, pyrocko); the `ml` extra adds the neural compression
and NPE training, `plotting` the posterior figures, `notebooks` Jupyter for the examples.

A modern conda (>= 23.10) solves this in minutes with the `libmamba` solver; on an older conda
run `conda install -n base conda-libmamba-solver` first, or append `--solver libmamba`.

Some notebooks and tests also need:

- a local Instaseis database, named by the `INSTASEIS_DB` environment variable (a 10 s PREM
  database is enough; the Azores notebook falls back to streaming `syngine://prem_i_2s`);
- [Computer Programs in Seismology](https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/index.html)
  (CPS), with `CPS_PATH` naming the directory holding `hprep96`, `hspec96` and `hpulse96`;
- `git lfs pull`, for the LV2 compression checkpoint.

## Getting started

The notebooks under `examples/` are the quickest way in. They run headless, and the
[documentation site](https://asaoulis.github.io/seismo-sbi/) renders them with their outputs.
Work through them in this order:

1. [`01_forward_models_and_receivers`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/01_forward_models_and_receivers.ipynb): receivers,
   the Instaseis, CPS and kernel forward models, a toy forward model plugged in through the
   registry, and the post-processing chain. Synthetic inputs; needs `INSTASEIS_DB` and `CPS_PATH`.
2. [`02_noise_covariances_and_likelihood`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/02_noise_covariances_and_likelihood.ipynb): every
   Gaussian-likelihood covariance on synthetic noise, the noise samplers, score compression, and
   MCMC with each covariance checked against the analytical posterior.
3. [`03_npe_training_and_evaluation`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/03_npe_training_and_evaluation.ipynb): dataset
   generation, training-time augmentation, training a neural compressor and flow, and the
   validation, calibration (TARP) and evaluation plots.
4. [`theory_errors_LV2`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/theory_errors_LV2.ipynb): theory-error SBI on the LV2 Long Valley
   Caldera event, against the Gaussian likelihood. Needs CPS and the git-lfs checkpoint.
5. [`nuisance_parameters_demo`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/nuisance_parameters_demo.ipynb): nuisance parameters drawn
   at simulation time, the source time function's duration and the post-processing effects
   (amplitude, dropout, time shift).
6. [`nuisance_augmentation_demo`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/nuisance_augmentation_demo.ipynb): the same effects
   applied in the dataloader as training-time augmentation.
7. [`azores_inversion`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/azores_inversion.ipynb): the 13/01/2022 Azores event of
   [Saoulis et al. (2024)](https://arxiv.org/abs/2410.23238), a fixed-location moment-tensor
   inversion with SBI and the Gaussian likelihood, then the full 10-parameter moment tensor,
   location and origin time.

The Azores notebook's first cell downloads and prepares its data:
```
cd scripts
python prepare_azores_example.py --output_dir ../examples/data/azores
```
This downloads the IPMA/CIVISA `PM`-network land-station data from IPMA's FDSN node (`http://ceida.ipma.pt`, the only open source for this network), removes the instrument response, filters and resamples, and writes the event waveform plus a few-hundred-window noise dataset under `examples/data/azores/`.

## How it works

Simulation-based inference (SBI) uses machine learning (ML) to build empirical models of key quantities in Bayesian inference. For example, SBI can train neural density estimators (NDEs) to build probabilistic models of the likelihood (which encodes a model of the data errors) or the posterior distribution explicitly. 

Seismic waveform data contains complicated noise and theory errors that common Gaussian likelihood assumptions fail to adequately model. This package uses the [`sbi`](https://github.com/sbi-dev/sbi) library to build, train, and sample from NDEs, which then serve as empirical surrogates of the likelihood. 

SBI builds a dataset of realistic observations, drawing samples from likelihood directly. It then trains a NDE to model the resulting likelihood (or posterior) distribution. Once trained, new observations can be fed through the NDE to perform inference, completely foregoing the forward model. 

![SBI Cartoon](assets/imgs/sbi_diagram.png)
_Fig. 3 from the `seismo-sbi` paper._

Forward modelling is currently performed using [`Instaseis`](https://instaseis.net/) and Computer Programmes for Seismology, though `seismo-sbi` is designed to be forward model agnostic. Every forward model lives in `seismo_sbi.simulators`; [docs/simulators.md](docs/simulators.md) maps the package and shows how to plug in your own. [docs/pipeline.md](docs/pipeline.md) describes a full inversion: the SBI pipeline, score compression and the Gaussian-likelihood inversion; [docs/training.md](docs/training.md) covers training neural compressors and amortised NPE models.

## Library map

Import each name from the module that defines it.

| package | what it holds |
|---|---|
| `simulators` | forward models: sources, receivers, the `Simulator` interface and its registry, Green's-function ensembles; backends in `instaseis/`, `cps/`, and `axisem/` (the perturbed 1-D Earth models an Instaseis ensemble is built from). See [docs/simulators.md](docs/simulators.md) |
| `nuisance_effects` | what a real recording does to a synthetic seismogram: amplitude, time-shift, dropout, scattering coda, anisotropy and dispersion effects, and the `PostProcessingChain` that applies them at simulation or training time |
| `sbi` | the inference pipeline (`pipeline`, `configuration`, `training_configuration`), dataset generation, scalers and job runners |
| `sbi.noises` | noise models: the Gaussian-likelihood covariances (diagonal, Toeplitz, theory-block), their estimator and samplers, and real-noise samplers for training |
| `sbi.compression` | compression to one summary per parameter: derivative stencils and score compressors; `ML/` holds the neural compressors and NPE training |
| `sbi.lsquares` | iterative least-squares source estimates |
| `sbi.types` | typed records passed between pipeline stages |
| `data_handling` | observed data ahead of inference; `preprocessing/` finds raw data, removes the response, filters, resamples, windows and writes the HDF5 the pipeline reads |
| `data_quality` | comparison of a reference synthetic with observed waveforms: per-trace metrics and the quality policy built on them |
| `priors` | catalogue-driven statistical priors for dataset generation |
| `evaluation` | pipeline build for evaluation, held-out validation and posterior metrics |
| `moment_tensor` | moment-tensor conventions, scalar moments, decompositions, pyrocko tensors, Kagan angles and lune angles |
| `plotting` | figures for simulations, posteriors and evaluation runs |
| `utils` | shared helpers: parallel execution, error handling, environment set-up, trace helpers |

The YAML configuration every pipeline is driven by is described in
[docs/configuration.md](docs/configuration.md).

## Command-line scripts

The tracked scripts under `scripts/` are launchers: each parses a few flags, builds the
configuration and calls one library entry point. Run any of them with `--help` for every flag.

| script | what it does | main flags |
|---|---|---|
| `train_NPE.py` | generates the training set and trains an NPE compressor and flow | `--config`, `--run-name`, `--stage {generate,meta,train}`, `--epochs`, `--devices`, `--architecture`, `--train-batch-size`, `--num-simulations` |
| `event_inversion.py` | runs a complete pipeline (Gaussian likelihood and/or SBI) on the synthetic and real events of a configuration | `--config` |
| `custom_download.py` | downloads waveforms and StationXML from FDSN providers | `--stations_file`, `--output_dir`, `--providers`, `--starttime`, `--endtime` |
| `build_catalogue.py` | builds event and noise HDF5 catalogues from downloaded data | see step 2 below |
| `custom_preprocess.py` | prepares one event and its noise without a full catalogue | `--data_dir`, `--output_dir`, `--event_name`, `--event_starttime`, `--event_endtime` |
| `prepare_azores_example.py` | downloads and prepares the data of the Azores example notebook | `--output_dir`, `--force` |
| `build_axisem_ensemble.py` | stages an AxiSEM ensemble of perturbed 1-D Earth models from one configuration | `--config`, `--dry-run`, `--from-bm-dir`, `--name` |

## Preparing real data

Preparing real seismic data for the SBI pipeline requires three steps: downloading raw waveforms and instrument responses, building event and noise h5 catalogues, and pointing the YAML config at the results.  All intermediate files are standard obspy formats (`.mseed` + StationXML); HDF5 is produced only at the final boundary step.

### 1. Download

Download BH? waveforms and StationXML for a time period long enough to include both your target events and a representative noise sample (weeks to months for a real study):

```bash
cd scripts
python custom_download.py \
    --stations_file configs/long_valley/stations.txt \
    --output_dir    /data/project \
    --starttime     2024-01-01T00:00:00 \
    --endtime       2024-02-01T00:00:00
```

Data are written to `{output_dir}/{station}/{year}.{jday}/` and StationXML to `{output_dir}/stationxml/`.

### 2. Build event + noise catalogues

`build_catalogue.py` does everything in one command: it pre-processes the raw data in daily chunks (response removal → filter → resample), then slices each event window and each clean noise window into an h5 file.  Pass either a QuakeML file or query FDSN directly:

```bash
# From a QuakeML catalogue file
python build_catalogue.py \
    --catalogue     events.xml \
    --data_dir      /data/project \
    --stationxml_dir /data/project/stationxml \
    --stations_file configs/long_valley/stations.txt \
    --output_dir    /data/catalogue \
    --duration      200 \
    --sampling_rate 1.0 \
    --noise_start   2024-01-01 \
    --noise_end     2024-02-01 \
    --n_jobs        8

# Or query FDSN for events automatically
python build_catalogue.py \
    --fdsn_query_center 35.7,-117.5 \
    --fdsn_min_magnitude 4.0 \
    --data_dir      /data/project \
    --stationxml_dir /data/project/stationxml \
    --stations_file configs/long_valley/stations.txt \
    --output_dir    /data/catalogue \
    --duration      200 --sampling_rate 1.0 \
    --noise_start   2024-01-01 --noise_end 2024-02-01 \
    --n_jobs        8
```

This produces:
```
/data/catalogue/events/{YYYYMMDDTHHMMSS}.h5   — one per target event
/data/catalogue/noise/{YYYY.MM.DD.HH.MM}.h5   — one per clean noise window
/data/catalogue/_daily/{station}/{YYYY.DDD}/  — cached daily processed mseed (reusable)
```

Each run is resumable: existing h5 and daily files are skipped automatically.

For a single event without a full noise catalogue, `custom_preprocess.py` is simpler:

```bash
python custom_preprocess.py \
    --data_dir      /data/project \
    --output_dir    /data/noise/long_valley \
    --event_name    LV2 \
    --event_starttime 1997-11-22T17:20:35 \
    --event_endtime   1997-11-22T17:23:54
```

### 3. Run the SBI inversion

In your YAML config:

- list each event file under `jobs.real_events` (a name mapped to an h5 path);
- to train on the recorded noise, set `inference.sbi.noise_model` to `type: 'real_noise'` with
  `noise_catalogue_path` naming the noise directory;
- to test the synthetic events against the same noise, also add `real_noise: <noise directory>` under
  `jobs.noise_models`;
- set `seismic_context.processing.filter_sampling_rate` to the raw rate the recordings were
  filtered at, so the synthetics are filtered at the same rate. It is required; see
  [docs/configuration.md](docs/configuration.md).

Then:

```bash
python event_inversion.py --config configs/long_valley/LV2_real.yaml
```

### HDF5 schema

Every h5 file produced by the pipeline has this layout, read directly by `RealNoiseSampler` and `SimulationDataLoader`:

```
/outputs/{station}/{Z,1,2}   — waveform arrays  (npts = compute_data_vector_length + 1)
/misc/{station}/{Z,1,2}      — autocorrelation from the pre-event noise window
```

Channel keys are always `Z`, `1`, `2` (never `E` or `N`).

## Testing

The test suite lives in `tests/` and uses [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for coverage.

| Marker | Description | Default |
|---|---|---|
| `unit` | Fast, isolated, no file I/O | Always run |
| `integration` | Loads data stubs (HDF5 fixtures built in `tmp_path`) | Always run |
| `slow` | Full end-to-end synthetic inversions — require Instaseis DB or CPS binaries | Skipped |

```bash
# Install test dependencies
pip install pytest pytest-cov

# Fast suite (unit + integration) — recommended after any edit
pytest tests/unit tests/integration -x -q

# Include slow end-to-end tests (need Instaseis DB or CPS installed)
pytest tests/ -x -q -m slow

# Full suite with coverage report
pytest tests/unit tests/integration \
    --cov=src/seismo_sbi \
    --cov-report=term-missing \
    --cov-report=html:htmlcov \
    -q
# then open htmlcov/index.html
```

The `slow` end-to-end tests exercise the full stencil→compression→inference pipeline and require at least one forward model:

- **Instaseis**: set `INSTASEIS_DB` to the path of a precomputed Green's function database (e.g. `PREM_10s`), or place it at `/data/shared/ROSA_PREM_10s_disc`.
- **CPS** (Computer Programs in Seismology): install the CPS suite so that `hprep96`, `hspec96`, and `hpulse96` are on `PATH`, or set `CPS_PATH` to the directory containing these binaries.

The `[test]` extras (`pytest`, `pytest-cov`) are declared in `pyproject.toml`. The repository's
workflow, `.github/workflows/docs.yml`, builds and publishes the documentation site on every push
to `main`; no workflow runs the test suite yet.

## Papers and citation

This package was used to produce the results in [Saoulis et al. (2025)](https://doi.org/10.1093/gji/ggaf112) and Saoulis et al. 2026 (in prep.). If you use it, please cite the relevant paper.

The data-errors example, [`azores_inversion`](https://github.com/asaoulis/seismo-sbi/blob/main/examples/azores_inversion.ipynb)
([Saoulis et al. (2025)](https://doi.org/10.1093/gji/ggaf112)), runs out of the box on the current
release (data download, processing and inversion). To reproduce the paper's full set of results
exactly, use the earlier release:

https://github.com/asaoulis/seismo-sbi/releases/tag/paper-release
