"""The pipeline plotter draws a job's corner plot, lunes and nodal-parameter plot from samples, and its
observed against synthetic traces."""
from pathlib import Path

import matplotlib
import numpy as np

from seismo_sbi.plotting.distributions import LUNE_REFERENCE_STYLES, MomentTensorReparametrised
from seismo_sbi.plotting.results_plotting import SBIPipelinePlotter
from seismo_sbi.plotting.seismo_plots import MisfitsPlotting
from seismo_sbi.sbi.configuration import SBI_Configuration
from seismo_sbi.sbi.scalers import FlexibleScaler
from seismo_sbi.sbi.types.results import InversionData, JobData
from seismo_sbi.simulators.receivers import Receivers

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


def test_the_reparametrised_corner_draws_each_reference_in_every_panel(tmp_path, monkeypatch):
    matplotlib.use("Agg")
    monkeypatch.chdir(EXAMPLES)
    parameters = SBI_Configuration.from_file("configs/npe_example.yaml").model_parameters
    rng = np.random.default_rng(0)
    truth = np.array([3.0, -2.0, -1.0, 1.5, -0.5, 2.0]) * 1e16
    references = {"agency A": truth * 1.05, "agency B": truth + 2e15}

    figure = MomentTensorReparametrised(None, parameters).plot_chain_consumer(
        {"posterior": (None, truth + rng.normal(scale=2e15, size=(400, 6)), None, None)},
        extra_references=references, kde=False, figsave=tmp_path / "corner.png")

    legend_labels = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
    assert legend_labels == ["posterior", "agency A", "agency B"]
    reference_colours = [matplotlib.colors.to_rgba(style["color"]) for style in LUNE_REFERENCE_STYLES[:2]]
    panels = [ax for ax in figure.axes if ax.get_visible() and len(ax.lines) == 0 and ax.collections]
    assert len(panels) == 15
    for ax in panels:
        drawn = [tuple(c.get_facecolor()[0]) for c in ax.collections
                 if len(c.get_offsets()) == 1 and len(c.get_facecolor())]
        assert all(colour in drawn for colour in reference_colours)


def test_vertical_only_misfits_are_one_figure(tmp_path):
    matplotlib.use("Agg")

    class _Parameters:
        def parameter_to_vector(self, key):
            return np.zeros(6)

    receivers = Receivers.from_arrays(["AAA", "BBB"], ["XX", "XX"], [1.0, 2.0], [0.0, 0.0])
    data_vector = np.random.default_rng(0).normal(size=2 * 3 * 300)
    job = JobData("job", "real_noise", data_vector, {})

    SBIPipelinePlotter(tmp_path, _Parameters()).plot_synthetic_misfits(
        job, receivers, 0.5 * data_vector, (0.0, 0.0), vertical_only=True)

    assert sorted(path.name for path in (tmp_path / "misfits").iterdir()) == ["stacked_job.png"]


def test_vertical_traces_are_scaled_to_the_largest_vertical_and_cut_to_the_window(monkeypatch):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "close", lambda *args: None)
    receivers = Receivers.from_arrays(["AAA", "BBB"], ["XX", "XX"], [1.0, 2.0], [0.0, 0.0])
    traces = np.random.default_rng(1).normal(size=(2, 3, 300)) * [[[1.0], [100.0], [100.0]]]
    data_vector = traces.reshape(-1)

    MisfitsPlotting(receivers, 1).plot_ordered_stacked_traces(data_vector, data_vector, (0.0, 0.0, 20),
                                                              components=("Z",), time_window_s=(0, 120))

    axis = plt.gcf().axes[0]
    offsets = np.repeat(axis.get_yticks(), 2)
    largest = max(np.abs(line.get_ydata() - offset).max() for line, offset in zip(axis.get_lines(), offsets))
    assert len(plt.gcf().axes) == 1 and np.isclose(largest, 1.0) and axis.get_xlim() == (0, 120)
