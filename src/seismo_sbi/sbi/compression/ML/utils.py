"""Checkpoint loading helpers.

:func:`unpickling_torch_load` is the context in which to load a checkpoint this library wrote,
hyperparameters included. Trusted files only.
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
