# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Affero General Public License v3, see <https://www.gnu.org/licenses/>.

"""Neural spline flow builder for the NPE density estimator.

:func:`build_nsf` stacks rational-quadratic spline coupling transforms, each followed by an LU
linear layer, on a standard-normal base conditioned on the embedding net's output.
:class:`ContextSplineMap` is the spline conditioner for a one-dimensional target, where
coupling has nothing else to condition on. Adapted from ``sbi``.
"""

from functools import partial

import torch.nn as nn
from nflows import transforms
from pyknos.nflows import distributions as distributions_
from pyknos.nflows import flows
from pyknos.nflows.nn import nets
from sbi.utils.torchutils import create_alternating_binary_mask
from torch import Tensor, relu, tensor, uint8


class ContextSplineMap(nn.Module):
    """
    Neural network from `context` to the spline parameters.

    We cannot use the resnet as conditioner to learn each dimension conditioned
    on the other dimensions (because there is only one). Instead, we learn the
    spline parameters directly. In the case of conditinal density estimation,
    we make the spline parameters conditional on the context. This is
    implemented in this class.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        context_features: int,
        hidden_layers: int,
    ):
        """Network predicting the spline parameters of a one-dimensional ``x`` from its context.

        :param in_features: unused, there being no conditioner in one dimension.
        :param out_features: number of spline parameters.
        :param hidden_features: hidden units.
        :param context_features: context features.
        """
        super().__init__()
        # `self.hidden_features` is only defined such that nflows can infer
        # a scaling factor for initializations.
        self.hidden_features = hidden_features

        # Use a non-linearity because otherwise, there will be a linear
        # mapping from context features onto distribution parameters.

        # Initialize with input layer.
        layer_list = [nn.Linear(context_features, hidden_features), nn.ReLU()]
        # Add hidden layers.
        layer_list += [
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
        ] * hidden_layers
        # Add output layer.
        layer_list += [nn.Linear(hidden_features, out_features)]
        self.spline_predictor = nn.Sequential(*layer_list)

    def __call__(self, inputs: Tensor, context: Tensor, *args, **kwargs) -> Tensor:
        """The spline parameters given the context.

        :param inputs: unused in one dimension.
        :param context: context features.
        :returns: the spline parameters.
        """
        return self.spline_predictor(context)


def build_nsf(
    dim : int,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
    embedding_net: nn.Module = nn.Identity(),
    tail_bound: float = 3.0,
    hidden_layers_spline_context: int = 1,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False,
    conditional_dim = None,
    use_identity = False,
    **kwargs,
) -> nn.Module:
    """Neural spline flow for ``p(x | y)``.

    :param batch_x: batch of ``x``, giving the dimensionality and the optional z-scoring.
    :param batch_y: batch of ``y``, likewise.
    :param z_score_x: ``None``/``'none'`` (no z-scoring), ``'independent'`` (per dimension) or
        ``'structured'`` (one mean and standard deviation over the whole batch, for a time series or
        an image).
    :param z_score_y: as ``z_score_x``, for ``y``.
    :param hidden_features: hidden features.
    :param num_transforms: number of transforms.
    :param num_bins: spline bins.
    :param embedding_net: optional embedding network for ``y``.
    :param tail_bound: tail bound of each spline.
    :param hidden_layers_spline_context: hidden layers of the spline context net for
        one-dimensional ``x``.
    :param num_blocks: residual blocks of the context embedding.
    :param dropout_probability: dropout probability in the residual net.
    :param use_batch_norm: batch norm in the residual net.
    :param kwargs: further build arguments, ignored.
    :returns: the flow.
    """
    x_numel = dim
    if conditional_dim:
        y_numel = conditional_dim
    else:
        y_numel = x_numel

    # Define mask function to alternate between predicted x-dimensions.
    def mask_in_layer(i):
        return create_alternating_binary_mask(features=x_numel, even=(i % 2 == 0))

    # If x is just a scalar then use a dummy mask and learn spline parameters using the
    # conditioning variables only.
    if x_numel == 1:
        # Conditioner ignores the data and uses the conditioning variables only.
        conditioner = partial(
            ContextSplineMap,
            hidden_features=hidden_features,
            context_features=y_numel,
            hidden_layers=hidden_layers_spline_context,
        )
    else:
        # Use conditional resnet as spline conditioner.
        conditioner = partial(
            nets.ResidualNet,
            hidden_features=hidden_features,
            context_features=y_numel,
            num_blocks=num_blocks,
            activation=relu,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
        )

    # Stack spline transforms.
    flow_constructor = transforms.PiecewiseRationalQuadraticCouplingTransform 
    transform_list = []
    for i in range(num_transforms):
        block = [
            flow_constructor(
                mask=mask_in_layer(i) if x_numel > 1 else tensor([1], dtype=uint8),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]
        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        if x_numel > 1:
            block.append(
                transforms.LULinear(x_numel, identity_init=True),
            )
        transform_list += block

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((x_numel,))
    neural_net = flows.Flow(transform, distribution, embedding_net)

    return neural_net
