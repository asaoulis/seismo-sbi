import torch
import math

from torch import nn

from .cnn_feature_extractor import ConvolutionalFeatureExtractor
from .csdi_transformer import ConditionalTransformer

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, ExponentialLR, StepLR

class SeismogramTransformer(nn.Module):

    def __init__(self, num_seismic_components, transformer_config, 
                        feature_length, num_outputs, noise_model,
                        seismogram_locations : torch.Tensor, device) -> None:
        super().__init__()

        
        self.feature_length = feature_length
        self.noise_model = noise_model
        

        self.CNN_feature_extractor = ConvolutionalFeatureExtractor(num_seismic_components, feature_length,
                                                                should_concat_location = True)
        
        self.all_station_transformer = ConditionalTransformer(feature_length, seismogram_locations, transformer_config, device=device)

        self.source_param_predictor = nn.Sequential(
        *[
            nn.Linear(feature_length, feature_length),
            nn.ReLU(),
            nn.Linear(feature_length, num_outputs)
        ]
        )
    
    def sample_noise_model(self, batch_size):
        return torch.stack([self.noise_model() for _ in range(batch_size)], dim =0)
    
    def forward(self, x : torch.Tensor):

        (batch_size, num_stations, num_seismic_components, trace_length) = x.shape
        x_flattened = x.reshape((batch_size*num_stations, num_seismic_components, trace_length))
        

        extracted_features = self.CNN_feature_extractor.forward(x_flattened)

        feature_sequences = extracted_features.reshape((batch_size, num_stations, self.feature_length))

        transformer_output = self.all_station_transformer(feature_sequences)

        aggregated_station_info = torch.mean(transformer_output, dim=1)

        return self.source_param_predictor (aggregated_station_info)

class LightningModel(pl.LightningModule):

    def __init__(self, loss_function = nn.MSELoss() , lr=0.001, **kwargs):

        super().__init__()

        self.model = SeismogramTransformer(**kwargs)
        self.loss_func = loss_function

        self.lr = lr
        self.learning_rate_sched = None

        self.weight_decay = 0

    def forward(self, x):
        return self.model.forward(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat , _= self.shared_step(batch)

        return y_hat, y
    
    def shared_step(self, batch, eval_type=""):

        x, y = batch
        noise = self.model.sample_noise_model(batch_size=x.shape[0])

        y_hat = self(x + noise)
        loss = self.loss_func(y_hat, y)
        loss_dict = {f'{eval_type}loss' : loss}

        return y_hat, loss_dict

    def training_step(self, batch, batch_idx):
        

        _, loss = self.shared_step(batch)


        self._log_loss(loss)
   
        return loss

    def validation_step(self, batch, batch_idx):

        _, loss = self.shared_step(batch, "val_")

        self._log_loss(loss)

        return loss

    def _log_loss(self, loss):
        
        for l in loss.keys():
            self.log(l, loss[l])

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """

        
        opt = torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate), weight_decay=self.weight_decay)
        if not self.learning_rate_sched:
            return opt
        if self.learning_rate_sched == 'one_cycle':
            print("Using one cycle LR: ", self.learning_rate_sched_opt)
            sch = OneCycleLR(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'reduce_on_plateau':
            sch = ReduceLROnPlateau(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'exponential':
            sch = ExponentialLR(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'step':
            sch = StepLR(opt, **self.learning_rate_sched_opt)
        sch = {"scheduler": sch, "interval": "epoch", "monitor": "val_loss"}
        return [opt], [sch]

class HybridPyroLightningModule(LightningModel):

    def __init__(self, model, pyro_loss_function, lr=0.001, **kwargs):
        self.compression_model = model

        self.lr = lr
        self.learning_rate_sched = None

        self.weight_decay = 0

        self.pyro_loss_function = pyro_loss_function
        self.pyro_model = pyro_loss_function.model
        self.guide = pyro_loss_function.guide
        self.lr = lr
        self.predictive = pyro_loss_function.infer.Predictive(
            self.pyro_model, guide=self.guide, num_samples=1
        )
    
    def forward(self, *args):
        return self.predictive(self.compression_model(*args))
    
    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        """

        
        opt = torch.optim.Adam({**self.parameters(),
                                **self.pyro_loss_function.parameters()}, 
                                lr=(self.lr or self.learning_rate), 
                                weight_decay=self.weight_decay)
        if not self.learning_rate_sched:
            return opt
        if self.learning_rate_sched == 'one_cycle':
            print("Using one cycle LR: ", self.learning_rate_sched_opt)
            sch = OneCycleLR(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'reduce_on_plateau':
            sch = ReduceLROnPlateau(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'exponential':
            sch = ExponentialLR(opt, **self.learning_rate_sched_opt)
        elif self.learning_rate_sched == 'step':
            sch = StepLR(opt, **self.learning_rate_sched_opt)
        sch = {"scheduler": sch, "interval": "epoch", "monitor": "val_loss"}
        return [opt], [sch]