"""Two code paths that could not run: the ML compressor's model loading and a coverage figure."""
import importlib

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytorch_lightning as pl
import torch

from seismo_sbi.plotting.coverage import plot_credibility_levels_histograms_dictionary
from seismo_sbi.sbi.compression.gaussian import MachineLearningCompressor


class TinyCompressor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


class IdentityScaler:
    def inverse_transform(self, x):
        return x


def test_the_ml_compressor_loads_its_best_checkpoint_and_compresses(tmp_path, monkeypatch):
    model = TinyCompressor()
    ckpts = tmp_path / "ml_models" / "tiny" / "ckpts"
    ckpts.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "pytorch-lightning_version": pl.__version__},
               ckpts / "tiny-epoch=03-val_loss=0.250000.ckpt")
    monkeypatch.chdir(tmp_path)
    compressor = MachineLearningCompressor(
        TinyCompressor, "tiny", lambda data: torch.as_tensor(data, dtype=torch.float32),
        IdentityScaler())
    data = np.arange(4.0)
    expected = model(torch.as_tensor(data, dtype=torch.float32).unsqueeze(0)).squeeze(0)
    assert torch.allclose(compressor.compress_data_vector(data), expected)


def test_the_credibility_histogram_figure_draws():
    # importing sbi.analysis (as the pipeline does) drops scienceplots' styles; register again
    importlib.reload(importlib.import_module("scienceplots"))
    alphas = np.linspace(0.0, 1.0, 11)
    coverage = (np.tile(alphas, (5, 1)) + 0.01 * np.arange(5)[:, None], alphas)
    plot_credibility_levels_histograms_dictionary({"run": coverage}, ["C0"])
