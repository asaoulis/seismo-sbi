## Full-waveform seismic source inversion using simulation-based inference

Improving moment tensor solutions by accounting for non-Gaussian data and theory errors in full waveform data using machine learning. 

This is the official repo used to produce the results in [Saoulis et al. (2025)](https://doi.org/10.1093/gji/ggaf112) and Saoulis et al. 2026 (in prep.).


### NEW: Theory errors example

After installation (see below), try running the minimal example to perform SBI on the LV2 SoCal Long Valley Caldera event:

[examples/theory_errors_LV2.ipynb](examples/theory_errors_LV2.ipynb)

### Data errors paper

The data-errors example, [examples/azores_inversion.ipynb](examples/azores_inversion.ipynb)
([Saoulis et al. (2025)](https://doi.org/10.1093/gji/ggaf112)), now runs out of the box on the
current `main` (data download, processing and inversion). We are still working towards an updated,
unified version of this repository, and some of the *other* example notebooks are not yet
back-compatible — to reproduce the full set of paper results you can still revert to the earlier
release:

https://github.com/asaoulis/seismo-sbi/releases/tag/paper-release

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
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

- Conda (Miniconda or Anaconda). A modern conda (>= 23.10) uses the fast `libmamba` solver by
  default; on an older conda the `instaseis` solve can take many minutes, so first run
  `conda install -n base conda-libmamba-solver` (or append `--solver libmamba` to the commands below).
- Python 3.11 (created for you by the environment file below; the current conda-forge `instaseis`
  requires Python >= 3.11).

### Installing

`instaseis` is historically the most fragile dependency, so **always start from a fresh environment
and install it first** — together with the rest of the scientific / seismology / geospatial stack —
from `conda-forge`. The provided [`environment.yml`](environment.yml) does exactly this; the
pure-Python ML / inference stack is then installed with `pip`:

```
conda env create -f environment.yml      # python 3.11 + instaseis + obspy + cartopy + basemap + ...
conda activate seismo-sbi
pip install -e .                          # torch, sbi, pytorch_lightning, pyrocko, ...
```

Equivalently, without the file:

```
conda create -n seismo-sbi -c conda-forge python=3.11 instaseis obspy "numpy>=2" "numba<0.62" scipy h5py matplotlib cartopy basemap pyproj
conda activate seismo-sbi
pip install -e .
```

Two example notebooks then run out of the box:

- [examples/theory_errors_LV2.ipynb](examples/theory_errors_LV2.ipynb) — theory-error SBI on the LV2
  Long Valley event. This one additionally needs **Computer Programs in Seismology (CPS)** installed
  (point the notebook's `CPS_PATH` at your install —
  https://www.eas.slu.edu/eqc/ComputerProgramsSeismology/index.html) and **git-lfs**
  (`git lfs pull`) to fetch the bundled compression checkpoint.
- [examples/azores_inversion.ipynb](examples/azores_inversion.ipynb) — see *Usage* below.

## Usage <a name = "usage"></a>

[examples/azores_inversion.ipynb](examples/azores_inversion.ipynb) uses SBI to perform (i) a fixed-location MT inversion and (ii) a full 10-parameter MT + time/location inversion for the 13/01/2022 Azores event in [Saoulis et al. (2024)](https://arxiv.org/abs/2410.23238). For (i), a comparison between SBI and the Gaussian likelihood approach is also provided, as it is computationally cheap.

The notebook's first cell downloads and prepares all of the data for you by running:
```
cd scripts
python prepare_azores_example.py --output_dir ../examples/data/azores
```
This downloads the IPMA/CIVISA `PM`-network land-station data from IPMA's FDSN node (`http://ceida.ipma.pt`, the only open source for this network), removes the instrument response, filters and resamples, and writes the event waveform plus a few-hundred-window noise dataset under `examples/data/azores/`.

Forward modelling uses a global PREM Instaseis database. By default the notebook streams it from IRIS Syngine (`syngine://prem_i_2s`) so it works anywhere; if you have a local database, set the environment variable `INSTASEIS_DB=/path/to/db` to use it instead (much faster, especially for the full inversion).
