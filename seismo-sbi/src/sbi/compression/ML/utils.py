from pathlib import Path

import torch
import numpy as np

from .seismogram_transformer import LightningModel

from pytorch_lightning.callbacks import ModelCheckpoint

def get_checkpoint_callback(name, save_top_k = -1, checkpoint_path = "model_ckpts", model=None):
    # filename template
    file_name = name + '-{epoch:02d}-{' + 'val_loss' + ':.6f}'
    path = Path(f'{checkpoint_path}/{name}/ckpts')
    
    if model:
        torch.save(model.state_dict(), path / "model.pt")
    # callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=path,
        filename=file_name,
        save_top_k=save_top_k
    )
    return checkpoint_callback
import re
def get_best_epoch(ckpts):
    exp = "(?<=val_loss=)(?:(?:\d+(?:\.\d*)?|\.\d+))"
    # print(ckpts)
    # # print([ckpt for ckpt in ckpts])
    # temp = next(ckpt)
    # print("test", temp)
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
            model = model_type.load_from_checkpoint(best_ckpt, **kwargs)
        ### TODO: Need to either catch a specific exception or re-raise the exception
        except Exception as e:
            print("Error loading model from checkpoint:\n", e)
            raise e
        if 'scaler' in kwargs:
            model.scaler = kwargs['scaler']
        return model