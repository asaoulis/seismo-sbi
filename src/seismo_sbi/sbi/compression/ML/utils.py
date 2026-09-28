"""Checkpoint loading helpers.

:func:`unpickling_torch_load` lets Lightning load this project's own checkpoints on torch 2.6 and
later, whose ``torch.load`` refuses the hyperparameters a checkpoint carries by default.
"""

import contextlib

import torch


@contextlib.contextmanager
def unpickling_torch_load():
    """Within the block ``torch.load`` also unpickles the hyperparameters a Lightning checkpoint carries. Trusted files only."""
    original_load = torch.load
    torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "weights_only": False})
    try:
        yield
    finally:
        torch.load = original_load
