"""The real-data width gate: posterior width on a fixed set of recordings, logged while training.

``RealWidthGate`` samples the flow on every recording of the set every few epochs and logs,
beside the validation loss, the fraction of events whose lune 95 % credible area is at most
``max_lune_area`` and the median area. ``load_observation_set`` reads the set from a directory
of ``.npz`` files written by ``write_observation``, one per event. ``gate_callbacks`` builds
the callback from the ``ml_real_width_gate`` configuration block.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytorch_lightning as pl

from seismo_sbi.moment_tensor.lune_angles import lune_credible_area, mts6_to_gamma_delta
from seismo_sbi.sbi.npe.posterior_sampling import sample_subsets_batched

#: Least number of finite draws an event needs for its lune area; with fewer the area is NaN,
#: which counts as not kept.
MIN_FINITE_DRAWS = 10


def write_observation(path, data, coords, source_vec=None, event_id: str = "") -> Path:
    """Save one recording for the gate: ``data`` ``(n_stations, n_components, n_samples)``,
    ``coords`` ``(n_stations, 2)`` latitude and longitude in degrees, and the source vector
    a conditioned model expects (``None`` for an unconditioned one)."""
    path = Path(path)
    np.savez(path, data=np.asarray(data, np.float32), coords=np.asarray(coords, np.float32),
             source_vec=np.asarray([] if source_vec is None else source_vec, np.float32),
             event_id=np.asarray(event_id or path.stem))
    return path


def load_observation_set(directory) -> tuple:
    """``(event ids, items)`` of every ``.npz`` under ``directory``, each item being the
    ``(data, coords, source_vec)`` triple the batched sampler takes."""
    ids, items = [], []
    for path in sorted(Path(directory).glob("*.npz")):
        with np.load(path) as saved:
            source_vec = saved["source_vec"]
            items.append((saved["data"], saved["coords"],
                          None if source_vec.size == 0 else source_vec))
            ids.append(str(saved["event_id"]))
    if not items:
        raise FileNotFoundError(f"no .npz observations under {directory}")
    return ids, items


def finite_lune_area(moment_tensors) -> float:
    """Lune 95 % credible area of the finite rows of ``moment_tensors`` ``(n_draws, >= 6)``,
    or NaN when fewer than ``MIN_FINITE_DRAWS`` rows are finite."""
    moment_tensors = np.asarray(moment_tensors, float)[:, :6]
    finite = moment_tensors[np.isfinite(moment_tensors).all(axis=1)]
    if len(finite) < MIN_FINITE_DRAWS:
        return float("nan")
    gamma, delta = mts6_to_gamma_delta(finite)
    return float(lune_credible_area(gamma, delta, 0.95))


class RealWidthGate(pl.Callback):
    """Log the kept fraction and median lune area of the flow on real recordings.

    ``items`` and ``event_ids`` come from :func:`load_observation_set`; ``data_scaler`` is the
    training-time parameter scaler whose ``inverse_transform`` returns physical moment
    tensors. The gate runs after validation every ``every_n_epochs`` epochs on rank 0, with
    ``num_samples`` draws per event, and keeps an event when its lune 95 % area is at most
    ``max_lune_area``. ``history`` holds ``(epoch, kept fraction, median area)`` per run.
    """

    def __init__(self, items, event_ids, data_scaler, *, every_n_epochs: int = 5,
                 num_samples: int = 500, max_lune_area: float = 0.25, prefix: str = "real"):
        super().__init__()
        if int(every_n_epochs) < 1:
            raise ValueError("every_n_epochs must be >= 1")
        self.items, self.event_ids, self.data_scaler = list(items), list(event_ids), data_scaler
        self.every_n_epochs, self.num_samples = int(every_n_epochs), int(num_samples)
        self.max_lune_area, self.prefix = float(max_lune_area), prefix
        self.history = []

    def lune_areas(self, pl_module) -> np.ndarray:
        """Lune 95 % credible area of the posterior on every item, in lune-area fraction; NaN
        for an item with fewer than ``MIN_FINITE_DRAWS`` finite draws."""
        posterior = SimpleNamespace(posterior_estimator=pl_module.flow, prior=None)
        was_training = pl_module.training
        pl_module.eval()
        samples = sample_subsets_batched(posterior, self.items, self.data_scaler,
                                         num_samples=self.num_samples, device=pl_module.device)
        if was_training:
            pl_module.train()
        return np.asarray([finite_lune_area(moment_tensors) for moment_tensors in samples], float)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or (trainer.current_epoch + 1) % self.every_n_epochs:
            return
        if trainer.global_rank != 0:
            return
        areas = self.lune_areas(pl_module)
        kept = float(np.mean(areas <= self.max_lune_area))
        median = float(np.nanmedian(areas))
        pl_module.log(f"{self.prefix}/kept_fraction", kept, rank_zero_only=True, sync_dist=False)
        pl_module.log(f"{self.prefix}/median_lune_area95", median, rank_zero_only=True, sync_dist=False)
        self.history.append((int(trainer.current_epoch), kept, median))
        print(f"[real width gate] epoch {trainer.current_epoch}: kept {kept:.2f} of "
              f"{len(areas)} events, median lune area {median:.3f}", flush=True)


def gate_callbacks(training, data) -> list:
    """The gate as a one-element callback list when ``ml_real_width_gate.observations`` names a
    directory, else an empty list."""
    block = training.real_width_gate
    if not block.observations:
        return []
    ids, items = load_observation_set(block.observations)
    return [RealWidthGate(items, ids, data.data_scaler, every_n_epochs=block.every_n_epochs,
                          num_samples=block.num_samples, max_lune_area=block.max_lune_area)]
