import torch

from torch import nn

class ConvolutionalFeatureExtractor(nn.Module):

    def __init__(self, num_seismic_components, final_feature_length, 
                     should_concat_location : bool, feedforward_layers = [256,256], **cnn_kwargs) -> None:
        super().__init__()

        self.seismic_trace_CNN = SeismicTraceCNN(num_seismic_components, **cnn_kwargs)
        self.feature_combination_operation = ConcatLayer() if should_concat_location else None  # TODO: Implement sinusoidal embeddings option 

        self.feedforward_net = FeedForwardFeatureProcessing(feedforward_layers, 
                                    output_dim = final_feature_length)


    def forward(self, x_trace):

        cnn_processed_trace = self.seismic_trace_CNN.forward(x_trace)
        return cnn_processed_trace
        return self.feedforward_net(cnn_processed_trace)


class ConcatLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, iterable_to_be_concatenated):
        return torch.cat(iterable_to_be_concatenated, dim = 1)


class SeismicTraceCNN(nn.Module):

    def __init__(
        self,
        num_seismic_components,
        conv_channels=None,          # list[int]: out_channels per conv
        conv_kernels=None,           # list[int]: kernel_size per conv
        conv_strides=None,           # list[int]: stride per conv
        same_padding=False,          # if True, uses kernel_size//2 padding
        activation="silu",           # "relu" | "gelu" | "silu"
        norm_type="batch",              # None | "batch" | "group" | "instance"
        dropout=0.1,                 # float in [0,1], applied after activation
    ) -> None:
        super().__init__()

        # Defaults that reproduce the previous hardcoded stack:
        # [Conv(num_comp->64, k=3,s=2), x4], then [Conv(64->16,k=3,s=1), Conv(16->4,k=3,s=1)]
        if conv_channels is None:
            conv_channels = [32, 64, 64, 128, 128, 128]
        if conv_kernels is None:
            conv_kernels = [5] * len(conv_channels)
        if conv_strides is None:
            # First 4 layers strided, rest stride=1 (matches previous behavior)
            n = len(conv_channels)
            n_strided = min(4, n)
            conv_strides = [2] * n_strided + [1] * (n - n_strided)

        if not (len(conv_channels) == len(conv_kernels) == len(conv_strides)):
            raise ValueError("conv_channels, conv_kernels, and conv_strides must have the same length")

        self._eps = 1e-8
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

        # Safe per-trace scaling to [-1,1] range by max abs value
        max_trace_val = x.abs().amax(dim=(1,2), keepdim=True).clamp_min(self._eps)
        scaled_x = x / max_trace_val

        scaled_x = self.conv_stack(scaled_x)

        scaled_x = self.flatten(scaled_x)
        # Concatenate log amplitude (avoid -inf)
        scaled_x = torch.cat([scaled_x, max_trace_val.clamp_min(self._eps).log()[:, 0]], dim = 1)

        return scaled_x


class FeedForwardFeatureProcessing(nn.Module):
    """
    Modern feedforward head:
    - Pre-norm MLP blocks (LayerNorm by default) to stabilize features coming from CNN.
    - Configurable activation (silu/gelu/relu) and dropout.
    - Residual connection when in/out dims match.
    - Optional output normalization (layer or L2) for stable embeddings.
    """

    def __init__(
        self,
        hidden_layer_dims,
        output_dim,
        activation: str = "silu",
        norm_type: str = "layer",      # "layer" | "batch" | None
        dropout: float = 0.1,
        residual: bool = True,
        output_norm: str = "layer" # None | "layer" | "l2"
    ) -> None:
        super().__init__()
        self._eps = 1e-8
        self._activation_name = (activation or "relu").lower()
        self._norm_type = (norm_type.lower() if norm_type is not None else None)
        self._dropout_p = float(dropout) if dropout else 0.0
        self._use_residual = bool(residual)
        self._output_norm = (output_norm.lower() if isinstance(output_norm, str) else None)

        # Build MLP blocks
        self.blocks = nn.ModuleList([
            _FFBlock(
                out_dim=h_dim,
                activation=self._get_activation(self._activation_name),
                norm=self._make_norm(self._norm_type),
                dropout_p=self._dropout_p,
                residual=self._use_residual,
            )
            for h_dim in (hidden_layer_dims or [])
        ])

        # Final projection head
        self.pre_out_norm = self._make_norm(self._norm_type)
        self.last_layer = nn.LazyLinear(output_dim)

        # Optional output normalization for stable embeddings
        self.out_layernorm = _LazyLayerNorm1d() if self._output_norm == "layer" else None

    def _get_activation(self, name: str) -> nn.Module:
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        return nn.ReLU()

    def _make_norm(self, norm_type: str ) -> nn.Module:
        if norm_type is None or norm_type == "none":
            return None
        if norm_type == "layer":
            return _LazyLayerNorm1d()
        if norm_type == "batch":
            # Works with 2D (N, C) feature tensors
            return nn.LazyBatchNorm1d()
        raise ValueError(f"Unsupported norm_type: {norm_type}")

    def _l2_normalize(self, x: torch.Tensor, dim: int = 1) -> torch.Tensor:
        return x / (x.norm(dim=dim, keepdim=True).clamp_min(self._eps))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        # Optional pre-out norm, then final projection
        if self.pre_out_norm is not None:
            x = self.pre_out_norm(x)
        x = self.last_layer(x)

        # Optional output normalization
        if self._output_norm == "layer" and self.out_layernorm is not None:
            x = self.out_layernorm(x)
        elif self._output_norm == "l2":
            x = self._l2_normalize(x, dim=1)

        return x


class _FFBlock(nn.Module):
    """
    Single pre-norm -> Linear -> Activation -> Dropout block with optional residual
    (residual applied only when in_features == out_features at runtime).
    """
    def __init__(
        self,
        out_dim: int,
        activation: nn.Module,
        norm: nn.Module,
        dropout_p: float,
        residual: bool,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.linear = nn.LazyLinear(out_dim)
        self.act = activation
        self.drop = nn.Dropout(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.use_residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x) if self.norm is not None else x
        y = self.linear(y)
        y = self.act(y)
        y = self.drop(y)
        if self.use_residual and y.shape[-1] == x.shape[-1]:
            y = y + x
        return y


class _LazyLayerNorm1d(nn.Module):
    """
    Lazy LayerNorm for 2D feature tensors (N, C); initializes on first forward.
    """
    def __init__(self, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        self._ln: nn.LayerNorm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._ln is None:
            self._ln = nn.LayerNorm(x.shape[-1], eps=self.eps, elementwise_affine=self.affine).to(x.device)
        return self._ln(x)