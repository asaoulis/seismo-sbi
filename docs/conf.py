"""Sphinx configuration: API pages from the package docstrings, landing page from the README."""
import shutil
from pathlib import Path

project = "seismo-sbi"
author = "Alex Saoulis"
extensions = ["autoapi.extension", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "myst_nb"]
autoapi_dirs = ["../src/seismo_sbi"]
autoapi_root = "api"
autoapi_options = ["members", "undoc-members", "show-inheritance", "show-module-summary"]
napoleon_use_ivar = True
myst_heading_anchors = 3
nb_execution_mode = "off"
suppress_warnings = ["myst.header"]
html_theme = "furo"
exclude_patterns = ["_build"]

examples = Path(__file__).parent / "_generated" / "examples"
examples.mkdir(parents=True, exist_ok=True)
for name in ("ridgecrest_obspy", "npe_flagship", "nuisances", "01_forward_models_and_receivers", "02_noise_covariances_and_likelihood",
             "03_npe_training_and_evaluation", "04_source_conventions", "05_resolution_and_tradeoffs",
             "theory_errors_LV2", "azores_inversion"):
    shutil.copy(Path(__file__).parents[1] / "examples" / f"{name}.ipynb", examples)
