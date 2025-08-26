import torch
from torch import nn
import os
from pathlib import Path
import re
from glob import glob
import numpy as np

from .seismogram_transformer import SeismogramTransformer, NPELightningModule
from .maf import build_nsf
from .dataloading import make_torch_dataloader, make_torch_dataloaders

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger  # added

class CompressionTrainer:

    def __init__(self, components, station_locations):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_seismic_components = len(components)
        feature_length = 128  # context dimension for the flow

        model_config = {"layers": 4,
            "channels": 128,
            "nheads": 4,
            "timeemb": 64,
            "posemb": 64}

        self.num_dims = 6  # dimensionality of theta
        latent_dim = 128
        self.latent_dim = latent_dim
        # Embedding network (context extractor)
        seismogram_transformer_model = SeismogramTransformer(
            num_seismic_components,
            model_config,
            feature_length,
            num_outputs=latent_dim,          # not used for embedding, but required by ctor
            noise_model=None,              # not used in NPE; pass None
            seismogram_locations=station_locations,
            device=self.device
        )


        # Conditional MAF over theta | x with embedding integrated in the flow
        self.flow = build_nsf(
            dim=self.num_dims,
            conditional_dim=latent_dim,
            hidden_features=128,
            num_transforms=5,
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=True,
            embedding_net=seismogram_transformer_model
        )

        # Lightning module that maximizes log p_phi(theta | x)
        self.model = NPELightningModule(
            flow=self.flow,
            lr=1e-4,
            weight_decay=1e-4,
        )

    def train(self, run_name, epochs=10, output_path=Path("model_ckpts"), dataloader_args: dict = None):
        if dataloader_args is None or "train_max_index" not in dataloader_args:
            raise ValueError("dataloader_args must include: data_loader, data_folder, parameter_name_map, synthetic_noise_model_sampler, and train_max_index.")

        # Build train/val dataloaders from a single split index
        train_dataloader, val_dataloader = make_torch_dataloaders(**dataloader_args)

        checkpoint_cb = create_best_checkpoint_callback(output_path / run_name)
        output_path = Path(output_path) / run_name
        wandb_logger = WandbLogger(project="seismo-sbi", name=output_path.parent.name + '/' + run_name)  # added

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            callbacks=[checkpoint_cb],
            precision=32,
            logger=wandb_logger,  # added
        )

        trainer.fit(self.model, train_dataloader, val_dataloader)

    def load_best(self, output_path: Path) -> Path:
        """
        Locate and load the best-performing checkpoint into self.model.
        Returns the path to the checkpoint that was loaded.
        """
        ckpt_path = find_best_checkpoint_path(output_path)
        print(ckpt_path)
        self.model = NPELightningModule.load_from_checkpoint(
            ckpt_path,
            flow=self.flow,
            lr=1e-4,
            weight_decay=1e-4,
        )
        self.model.eval()
        self.model.freeze()
        return ckpt_path
        
    
    def build_posterior(self):
        # use sbi to build a direct posterior from the trained flow
        from sbi.inference.posteriors import DirectPosterior
        from sbi import utils as utils

        prior = utils.BoxUniform(low=np.zeros((self.num_dims)), high=np.ones((self.num_dims)), device='cuda')

        posterior = DirectPosterior(
                    posterior_estimator=self.model.flow.to('cuda'),
                    prior=prior,
                    # x_shape=self._x_shape,
                    device='cuda',
                )
        return posterior
from pytorch_lightning.callbacks import ModelCheckpoint

def create_best_checkpoint_callback(output_path):
    best_checkpoint_callback = ModelCheckpoint(
            dirpath=f'{output_path}/checkpoints',
            filename='best_model-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
    return best_checkpoint_callback

def find_best_checkpoint_path(output_path: Path) -> Path:
    """
    Find the best .ckpt under <output_path>/checkpoints by parsing the val_loss
    encoded in the filename 'best_model-{val_loss:.2f}.ckpt'.
    """
    ckpt_dir = Path(output_path) / "checkpoints"
    candidates = list(ckpt_dir.glob("best_model-*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    def score_from_name(p: Path) -> float:
        m = re.search(r"best_model-val_loss=(-?\d+(?:\.\d+)?)(?:-v\d+)?\.ckpt$", p.name)
        return float(m.group(1)) if m else float("inf")
    best = min(candidates, key=score_from_name)
    return best