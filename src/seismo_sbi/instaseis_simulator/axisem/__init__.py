"""AxiSEM ensemble producer: read/write/perturb external 1-D background models
(``background_model.bm``) and generate ensembles of perturbed Earth models for
theory-error Green's-function databases.

The consumer side (``InstaseisEnsembleSimulator`` in
``seismo_sbi.instaseis_simulator.ensemble``) ingests the repacked Instaseis
databases this producer ultimately yields on the cluster.
"""

from .model_io import BackgroundModel, read_bm, write_bm
from .perturb import perturb_background_model
from .ensemble import generate_ensemble

__all__ = [
    "BackgroundModel",
    "read_bm",
    "write_bm",
    "perturb_background_model",
    "generate_ensemble",
]
