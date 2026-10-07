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
myst_enable_extensions = ["dollarmath", "amsmath"]
myst_heading_anchors = 3
nb_execution_mode = "off"
suppress_warnings = ["myst.header"]
html_theme = "furo"
exclude_patterns = ["_build"]

examples = Path(__file__).parent / "_generated" / "examples"
shutil.rmtree(examples, ignore_errors=True)
examples.mkdir(parents=True)
for name in ("ridgecrest_obspy", "npe_flagship", "nuisances", "custom_forward_model", "02_noise_covariances_and_likelihood",
             "theory_errors_LV2", "azores_inversion"):
    shutil.copy(Path(__file__).parents[1] / "examples" / f"{name}.ipynb", examples)
shutil.copytree(Path(__file__).parents[1] / "examples" / "assets", examples / "assets")
