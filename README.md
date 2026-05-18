## Full-waveform seismic source inversion using simulation-based inference

Improving moment tensor solutions by accounting for non-Gaussian data and theory errors in full waveform data using machine learning. 

This is the official repo used to produce the results in [Saoulis et al. (2025)](https://doi.org/10.1093/gji/ggaf112) and Saoulis et al. 2026 (in prep.).


### NEW: Theory errors example

After installation (see below), try running the minimal example to perform SBI on the LV2 SoCal Long Valley Caldera event:

[examples/theory_errors_LV2.ipynb](examples/theory_errors_LV2.ipynb)

### Data errors paper

We are currently working on an updated, unified version of this repository. However, some example notebooks are not backware compatible yet. For the data errors paper [Saoulis et al. (2025)](https://doi.org/10.1093/gji/ggaf112), revert to the earlier release to ensure all examples work correctly:

https://github.com/asaoulis/seismo-sbi/releases/tag/paper-release

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Testing](#testing)
- [Technical Details](#technical)

## About <a name = "about"></a>

Simulation-based inference (SBI) uses machine learning (ML) to build empirical models of key quantities in Bayesian inference. For example, SBI can train neural density estimators (NDEs) to build probabilistic models of the likelihood (which encodes a model of the data errors) or the posterior distribution explicitly. 

Seismic waveform data contains complicated noise and theory errors that common Gaussian likelihood assumptions fail to adequately model. This package uses the [`sbi`](https://github.com/sbi-dev/sbi) library to build, train, and sample from NDEs, which then serve as empirical surrogates of the likelihood. 

Forward modelling is currently performed using [`Instaseis`](https://instaseis.net/) and Computer Programmes for Seismology, though `seismo-sbi` is designed to be forward model agnostic.

### `seismo-sbi` workflow

SBI builds a dataset of realistic observations, drawing samples from likelihood directly. It then trains a NDE to model the resulting likelihood (or posterior) distribution. Once trained, new observations can be fed through the NDE to perform inference, completely foregoing the forward model. 

![SBI Cartoon](assets/imgs/sbi_diagram.png)
_Fig. 3 from the `seismo-sbi` paper._

## Getting Started <a name = "getting_started"></a>

### Prerequisites

- Conda
- Python >=3.8

Install Anaconda or Miniconda. Set up your conda environment by executing the following command in the terminal, assuming `my_env` is the name of your conda environment:

```
conda create -n "my_env" python=3.8
conda activate my_env
```

### Installing

First, install `instaseis`, which is best installed through `conda-forge`:

```
conda install -y -c conda-forge instaseis
```

Installation of the library can then be done by navigating to the top-level directory `seismo-sbi` and running:
```
pip install -e .
```

## Usage <a name = "usage"></a>

An example notebook is provided under [examples/azores_inversion.ipynb](examples/azores_inversion.ipynb). This notebook uses SBI to perform a (i) fixed location MT inversion and (ii) full 10-parameter MT and time-location for the 13/01/2022 Azores event in [Saoulis et al. (2024)](https://arxiv.org/abs/2410.23238). For (i), a comparison between SBI and the Gaussian likelihood approach is provided as it is computationally cheap.

Before running the notebook, you will need to run the two provided scripts
```
cd scripts
python download.py
python generate_noise_database.py
```
which downloads the nearby IPMA permanent land station data, and then processes the data to build an event file and a noise catalogue.

## Testing <a name = "testing"></a>

The test suite lives in `tests/` and uses [pytest](https://docs.pytest.org/) with [pytest-cov](https://pytest-cov.readthedocs.io/) for coverage.

### Test markers

| Marker | Description | Default |
|---|---|---|
| `unit` | Fast, isolated, no file I/O | Always run |
| `integration` | Loads data stubs (HDF5 fixtures built in `tmp_path`) | Always run |
| `slow` | Full end-to-end synthetic inversions — require Instaseis DB or CPS binaries | Skipped |

### Running the tests

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

### Slow test requirements

The `slow` end-to-end tests exercise the full stencil→compression→inference pipeline and require at least one forward model:

- **Instaseis**: set `INSTASEIS_DB` to the path of a precomputed Green's function database (e.g. `PREM_10s`), or place it at `/data/shared/ROSA_PREM_10s_disc`.
- **CPS** (Computer Programs in Seismology): install the CPS suite so that `hprep96`, `hspec96`, and `hpulse96` are on `PATH`, or set `CPS_PATH` to the directory containing these binaries.

### GitHub Actions

A minimal CI workflow runs the fast suite on every push.  Add `.github/workflows/tests.yml`:

```yaml
name: tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: conda-incubator/setup-miniconda@v3
        with:
          python-version: "3.8"
          channels: conda-forge,defaults
      - name: Install dependencies
        run: |
          conda install -y -c conda-forge instaseis
          pip install -e ".[test]"
      - name: Run fast tests
        run: pytest tests/unit tests/integration -x -q --cov=src/seismo_sbi --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
```

Add `[test]` extras to `pyproject.toml` if not already present:

```toml
[project.optional-dependencies]
test = ["pytest", "pytest-cov"]
```
