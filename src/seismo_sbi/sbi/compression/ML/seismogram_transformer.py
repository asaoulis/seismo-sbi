import torch
import math

from torch import nn

from .cnn_feature_extractor import ConvolutionalFeatureExtractor
from .csdi_transformer import ConditionalTransformer
from .axial_transformer import SeismogramAxialTransformer

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, ExponentialLR, StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LambdaLR, CyclicLR

import torch
import torch.nn as nn
import torch.nn.functional as F

class SeismogramTransformer(nn.Module):

    def __init__(self, num_seismic_components, transformer_config, 
                        feature_length, num_outputs, noise_model,
                        seismogram_locations : torch.Tensor, device, 
                        aggregation: str = "mean") -> None:
        super().__init__()

        self.feature_length = feature_length
        self.noise_model = noise_model

        # Validate aggregation choice
        if aggregation not in ("mean", "query"):
            raise ValueError(f"Invalid aggregation '{aggregation}'. Choose 'mean' or 'query'.")
        self.aggregation = aggregation
        d_model = transformer_config['channels']


        self.CNN_feature_extractor = ConvolutionalFeatureExtractor(
            num_seismic_components, cnn_output_dim=d_model, final_feature_length=feature_length,
            should_concat_location = True
        )
        # self.all_station_transformer = ConditionalTransformer(
        #     feature_length, seismogram_locations, transformer_config, device=device
        # )
        mode = 'axial'
        self.L = self.CNN_feature_extractor.seismic_trace_CNN.output_length
        self.D = self.CNN_feature_extractor.seismic_trace_CNN.output_channels  
        # New: allow configuring pooling and CLS from transformer_config
        pool_method = transformer_config.get("pooling", "mean")  # supports: mean, first, attn, max, gem
        use_cls = transformer_config.get("use_cls_token", False)
        num_q = transformer_config.get("num_query_tokens", 8)
        self.all_station_transformer = SeismogramAxialTransformer(
            seismogram_locations,
            d_model=d_model,
            # d_model=transformer_config['channels'],
            nheads=transformer_config["nheads"],
            num_layers=transformer_config["layers"],
            time_steps=self.L,
            conv_length=self.D,
            mode=mode,
            pool_queries=pool_method,
            use_cls_token=use_cls,
            num_query_tokens=num_q,
            temporal_pool_tokens=transformer_config.get("temporal_pool_tokens", 0),   # New
        )

        self.source_param_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_outputs)
        )
    
    def sample_noise_model(self, batch_size):
        return torch.stack([self.noise_model() for _ in range(batch_size)], dim =0)
    
    def forward(self, x : torch.Tensor):
        aggregated_station_info = self.embed(x)
        outputs = self.source_param_predictor(aggregated_station_info)
        return outputs

    def embed(self, x: torch.Tensor):
        (batch_size, num_stations, num_seismic_components, trace_length) = x.shape
        B, N = batch_size, num_stations

        # flatten to feed CNN
        x_flattened = x.reshape((batch_size*num_stations, num_seismic_components, trace_length))
        extracted_features = self.CNN_feature_extractor(x_flattened)
        # reshape back to (B, N, L, D)
        feature_sequences = extracted_features.view(B, N, self.D, self.L).permute(0, 1, 3, 2).contiguous()
        # contextualize with transformer
        transformer_output = self.all_station_transformer(feature_sequences)  # (x, q, pooled)
        # Aggregate across stations based on selected operator
        return transformer_output[2] # pooled embedding


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

class NPELightningModule(pl.LightningModule):
    def __init__(self, flow, lr=1e-3, weight_decay=0.0, **kwargs):
        super().__init__()
        self.flow = flow
        self.lr = lr
        self.weight_decay = weight_decay
        # New: allow selecting second-stage scheduler ("cosine" or "cyclic")
        self.lr_second_stage = "cosine"
        self.cyclic_period_steps = 8000

    def forward(self, x, theta):
        # The flow contains the embedding_net; pass x as context to be embedded internally.
        return self.flow.log_prob(theta, context=x)

    def training_step(self, batch, batch_idx):
        theta, x = batch
        log_prob = self.forward(x, theta)
        loss = -log_prob.mean()
        self.log("loss", loss, prog_bar=True)
        self.log("log_prob", log_prob.mean())
        return loss

    def validation_step(self, batch, batch_idx):
        theta, x = batch
        log_prob = self.forward(x, theta)
        val_loss = -log_prob.mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_log_prob", log_prob.mean())
        return val_loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Default: 2-phase schedule → warmup → cosine or cyclic
        max_epochs = getattr(self.trainer, "max_epochs", None) or 500

        warmup_epochs = max(1, int(0.05 * max_epochs))
        cosine_epochs = max_epochs - warmup_epochs  # remaining epochs

        # If using cyclic LR, switch to step-based scheduling and compute warmup in steps
        use_cyclic = (str(self.lr_second_stage).lower() == "cyclic")

        if use_cyclic:
            # Try to derive steps per epoch
            total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if total_steps is not None and max_epochs > 0:
                steps_per_epoch = max(1, total_steps // max_epochs)
            else:
                steps_per_epoch = getattr(self.trainer, "num_training_batches", None) or 1

            warmup_steps = max(1, warmup_epochs * steps_per_epoch)

            # Linear warmup to base LR over warmup_steps
            def lr_lambda_warmup(step_idx):
                return min(float(step_idx + 1) / float(max(1, warmup_steps)), 1.0)
            warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

            # Cyclic LR with given period (in steps)
            half_period = max(1, self.cyclic_period_steps // 2)
            cyclic = CyclicLR(
                optimizer,
                base_lr=self.lr * 0.05,
                max_lr=self.lr,
                step_size_up=half_period,
                step_size_down=self.cyclic_period_steps - half_period,
                mode="triangular",
                cycle_momentum=False
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cyclic],
                milestones=[warmup_steps],
            )

            return [optimizer], [{
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }]

        # Linear warmup to base LR (epoch-based)
        def lr_lambda_warmup(epoch):
            return float(epoch + 1) / float(max(1, warmup_epochs))
        warmup = LambdaLR(optimizer, lr_lambda=lr_lambda_warmup)

        # Cosine annealing down to 10% of base LR
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=self.lr * 0.1)

        # Combine: warmup → cosine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }]
