
import torch

from .seismogram_transformer import SeismogramTransformer, LightningModel, HybridPyroLightningModule
from .maf import create_masked_autoregressive_flow

from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoNormal


class CompressionTrainer:

    def __init__(self, components, station_locations):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        num_seismic_components = len(components)
        num_features_per_station = 128

        model_config = {"layers": 4,
            "channels": 64,
            "nheads": 8,
            "timeemb": 64,
            "posemb": 64}


        self.seismogram_transformer_model = SeismogramTransformer(num_seismic_components, model_config, num_features_per_station, 10, station_locations,device=self.device)
        self.neural_posterior_estimator = create_masked_autoregressive_flow(10)

        guide = AutoNormal(self.neural_posterior_estimator)
        pyro_loss_fn = Trace_ELBO()(self.neural_posterior_estimator, guide)
        self.model = HybridPyroLightningModule(self.seismogram_transformer_model, pyro_loss_fn, lr =0.0001)


    def train(self, train_dataloader, valid_dataloader, epochs=10):
        # All relevant parameters need to be initialized before ``configure_optimizer`` is called.
        # Since we used AutoNormal guide our parameters have not be initialized yet.
        # Therefore we initialize the model and guide by running one mini-batch through the loss.
        mini_batch = next(iter(train_dataloader))
        self.loss_fn(*mini_batch)
        wandb_logger = WandbLogger(project='instaseis-sbi', name=run_name)

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator=self.device,
            logger=wandb_logger,
            callbacks=[best_checkpoint_callback, plot_callback]  # Add the checkpoint callback
        )  # Use GPU if available, otherwise use CPU

        trainer.fit(self.model, train_dataloader, valid_dataloader)
        wandb.finish()

from pytorch_lightning.callbacks import ModelCheckpoint

def create_best_checkpoint_callback(output_path):
    best_checkpoint_callback = ModelCheckpoint(
            dirpath=f'{output_path}/checkpoints',
            filename='best_model-{val_loss:.2f}',
            save_top_k=3,
            monitor='val_loss',
            mode='min'
        )
    retunr best_checkpoint_callback