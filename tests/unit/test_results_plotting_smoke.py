"""The pipeline plotter draws a job's corner plot, lunes and nodal-parameter plot from samples."""
from pathlib import Path

import matplotlib
import numpy as np

from seismo_sbi.plotting.results_plotting import SBIPipelinePlotter
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.scalers import FlexibleScaler
from seismo_sbi.sbi.types.results import InversionData

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_chain_consumer_figures_from_synthetic_samples(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    monkeypatch.chdir(EXAMPLES)
    parameters = SBI_Configuration.from_file("configs/npe_example.yaml").model_parameters
    scaler = FlexibleScaler(parameters)
    rng = np.random.default_rng(0)
    truth = np.array([3.0, -2.0, -1.0, 1.5, -0.5, 2.0]) * 1e16
    samples = {name: InversionData(truth, truth + rng.normal(scale=2e15, size=(400, 6)), scaler)
               for name in ("first", "second")}

    plotter = SBIPipelinePlotter(tmp_path, parameters)
    plotter.initialise_posterior_plotter(scaler, parameters.parameter_to_vector("information")[:6])
    plotter.plot_chain_consumer("inversions", "job", samples, kde=False)

    written = sorted(path.name for path in (tmp_path / "inversions").iterdir())
    assert written == ["job.svg", "lune_job.svg", "lune_kde_job.svg", "nodal_params_job.svg"]
    assert all((tmp_path / "inversions" / name).stat().st_size > 1000 for name in written)
