
from nflows import distributions as distributions_
from nflows import flows, transforms
from torch import Tensor, nn, relu, tanh

def create_masked_autoregressive_flow(
    input_dim: Tensor,
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_blocks: int = 2,
    dropout_probability: float = 0.0,
    use_batch_norm: bool = False
    ):

    transform_list = []
    for _ in range(num_transforms):
        block = [
            transforms.MaskedAffineAutoregressiveTransform(
                features=input_dim,
                hidden_features=hidden_features,
                num_blocks=num_blocks,
                use_residual_blocks=False,
                random_mask=False,
                activation=tanh,
                dropout_probability=dropout_probability,
                use_batch_norm=use_batch_norm,
            ),
            transforms.RandomPermutation(features=input_dim),
        ]
        transform_list += block

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)

    distribution = distributions_.StandardNormal((input_dim,))
    neural_net = flows.Flow(transform, distribution)

    return neural_net