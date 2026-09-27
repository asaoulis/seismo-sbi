"""Checkpoint lookup for the regression compressor.

:func:`get_best_epoch` picks the checkpoint with the lowest ``val_loss`` in its filename;
:func:`get_best_model` loads it into a ``LightningModel``. :func:`unpickling_torch_load` lets
Lightning load this project's own checkpoints on torch 2.6 and later.
"""

import contextlib
from pathlib import Path

import numpy as np
import torch

from .seismogram_transformer import LightningModel


import re
def get_best_epoch(ckpts):
    exp = "(?<=val_loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    losses = [float(re.findall(exp, ckpt.name)[0]) for ckpt in ckpts]
    if len(losses) == 0:
        raise ValueError("No checkpoints found")
    ckpt = ckpts[np.argmin(losses)]
    print("Using checkpoint", ckpt, "\n")
    return ckpt

def get_best_model(model_type : LightningModel, name, 
                    checkpoint_path = "model_ckpts",*args, **kwargs) -> LightningModel:

    model_ckpts_dir = Path(f'{checkpoint_path}/{name}/ckpts')
    print(model_ckpts_dir)
    ckpts = list(model_ckpts_dir.glob('**/*.ckpt'))

    if len(list(ckpts)) == 0:
        print("No checkpoint found")
        return None
    else:
        best_ckpt = get_best_epoch(ckpts)
        print("Loading model from checkpoint", best_ckpt, "\n")
        try:
            with unpickling_torch_load():
                model = model_type.load_from_checkpoint(best_ckpt, **kwargs)
        except Exception as e:
            print("Error loading model from checkpoint:\n", e)
            raise e
        if 'scaler' in kwargs:
            model.scaler = kwargs['scaler']
        return model


@contextlib.contextmanager
def unpickling_torch_load():
    """Within the block ``torch.load`` also unpickles the hyperparameters a Lightning checkpoint carries. Trusted files only."""
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "weights_only": False})
    try:
        yield
    finally:
        torch.load = original_load
