"""Per-station 1-D convolutional trace encoder.

:class:`SeismicTraceCNN` normalises each trace ``(B, C, T)`` by its peak amplitude, runs a
configurable Conv1d stack over the components, and appends the log peak amplitude as an extra
channel. ``station_encoders.CNNEncoder`` wraps it as the default station encoder.
"""

import torch

from torch import nn

from seismo_sbi.sbi.npe.networks.station_encoders import normalize_trace, log_amp_channel


class SeismicTraceCNN(nn.Module):

    def __init__(
        self,
        num_seismic_components,
        input_length=200,
        final_layer=128,
        conv_channels=None,          # list[int]: out_channels per conv
        conv_kernels=None,           # list[int]: kernel_size per conv
        conv_strides=None,           # list[int]: stride per conv
        same_padding=False,          # if True, uses kernel_size//2 padding
        downsample=None,             # int: total temporal stride (single strided front conv,
                                     #   same-padded rest). Uniform with tcn/pno; makes the stack
                                     #   length-predictable + safe on short (e.g. decimated) inputs.
        activation="silu",           # "relu" | "gelu" | "silu"
        norm_type="batch",              # None | "batch" | "group" | "instance"
        dropout=0.1,                 # float in [0,1], applied after activation
    ) -> None:
        super().__init__()

        # Defaults that reproduce the previous hardcoded stack:
        # [Conv(num_comp->64, k=3,s=2), x4], then [Conv(64->16,k=3,s=1), Conv(16->4,k=3,s=1)]
        if conv_channels is None:
            conv_channels = [64, 64, 64, 128, 128, final_layer-1]
        if conv_kernels is None:
            conv_kernels = [5] * len(conv_channels)
        # ``downsample``: one strided front conv, all else stride 1 and same-padded, so the output is
        # ceil(input_length / downsample) long; it overrides the multi-strided default.
        if downsample is not None:
            same_padding = True
            n = len(conv_channels)
            conv_strides = [int(downsample)] + [1] * (n - 1)
        if conv_strides is None:
            # First 2 layers strided, rest stride=1 (legacy behavior)
            n = len(conv_channels)
            n_strided = min(2, n)
            conv_strides = [2] * n_strided + [1] * (n - n_strided)

        # Compute output temporal length after the full conv stack
        L = int(input_length)
        for k, s in zip(conv_kernels, conv_strides):
            pad = (k // 2) if same_padding else 0
            # Conv1d output length: floor((L + 2*pad - (k - 1) - 1)/stride + 1)
            L = (L + 2 * pad - (k - 1) - 1) // s + 1
            if L <= 0:
                raise ValueError(
                    f"Conv stack produces non-positive length after a layer: "
                    f"input_length={input_length}, k={k}, stride={s}, pad={pad}, current_L={L}. "
                    f"Check conv_kernels/strides or increase input_length."
                )
        self.output_length = L
        self.output_channels = conv_channels[-1] + 1  # +1 for log amplitude

        if not (len(conv_channels) == len(conv_kernels) == len(conv_strides)):
            raise ValueError("conv_channels, conv_kernels, and conv_strides must have the same length")

        self._eps = 1e-12
        in_c = num_seismic_components
        layers = []
        for out_c, k, s in zip(conv_channels, conv_kernels, conv_strides):
            pad = (k // 2) if same_padding else 0
            layers.append(nn.Conv1d(in_c, out_c, kernel_size=k, stride=s, padding=pad))
            norm_layer = self._get_norm(norm_type, out_c)
            if norm_layer is not None:
                layers.append(norm_layer)
            layers.append(self._get_activation(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            in_c = out_c

        self.conv_stack = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def _get_activation(self, name: str) -> nn.Module:
        name = (name or "relu").lower()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        # default
        return nn.ReLU()

    def _get_norm(self, norm_type: str , num_features: int) -> nn.Module:
        if norm_type is None:
            return None
        t = norm_type.lower()
        if t == "batch":
            return nn.BatchNorm1d(num_features)
        if t == "instance":
            return nn.InstanceNorm1d(num_features, affine=True)
        if t == "group":
            # Use 8 groups when possible; fall back to 1 (InstanceNorm-like) if small
            groups = 8 if num_features >= 8 else 1
            return nn.GroupNorm(num_groups=groups, num_channels=num_features)
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    def forward(self, x):

        # Safe per-trace scaling to [-1,1] range by max abs value (shared helper).
        scaled_x, max_trace_val = normalize_trace(x, self._eps)

        scaled_x = self.conv_stack(scaled_x)

        # Append the broadcast log-amplitude channel (avoids -inf via clamp).
        amp_feature = log_amp_channel(max_trace_val, scaled_x.size(-1), self._eps)  # (B,1,L)
        scaled_x = torch.cat([scaled_x, amp_feature], dim=1)
        return scaled_x
