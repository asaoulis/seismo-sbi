import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class SkipTransformer(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, nheads):
        super().__init__()
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info):
        B, channel, K, L = x.shape
        base_shape = x.shape
        y = x.reshape(B, channel, K * L)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip



class ConditionalTransformer(nn.Module):
    def __init__(self, seq_len_in, station_positions, config, device, *args, **kwargs):
        super().__init__()
        self.device = device
        self.timepoints = torch.Tensor(np.arange(seq_len_in)).to(self.device)
        self.station_positions = station_positions.to(self.device)

        self.emb_time_dim = config["timeemb"]
        self.emb_position_dim = config["posemb"]

        self.emb_total_dim = self.emb_time_dim  + self.emb_position_dim

        config["side_dim"] = self.emb_total_dim

        input_dim = 1
        self.diffmodel = SkipTransformer(config, input_dim)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def station_position_embedding(self, pos, d_model=128):
        pes = []
        for coord_index in range(2):
            pe = torch.zeros(pos.shape[0], pos.shape[1], d_model//2).to(self.device)
            position = pos[:, :, coord_index]
            position = position.unsqueeze(2).repeat(1, 1,d_model//4)
            div_term = 1 / torch.pow(
                10000.0, torch.arange(0, d_model//2, 2).to(self.device) / (d_model//2)
            )
            div_term = div_term.unsqueeze(0).repeat(pos.shape[1], 1)
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            pes.append(pe)
        return torch.cat(pes, dim=-1)

    def get_side_info(self, X):
        B, K, L = X.shape

        timepoints = self.timepoints.repeat(B,1)
        time_embed = self.time_embedding(timepoints, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)

        station_positions = self.station_positions.repeat(B,1,1)
        pos_embed = self.station_position_embedding(station_positions, self.emb_position_dim)  # (B,K,emb)
        pos_embed = pos_embed.unsqueeze(1).expand(-1, L, -1, -1)

        side_info = torch.cat([time_embed, pos_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

 
    def forward(self, batch):
        X = batch

        side_info = self.get_side_info(X)

        predicted = self.diffmodel(X.unsqueeze(1), side_info)  # (B,K,L)

        return predicted