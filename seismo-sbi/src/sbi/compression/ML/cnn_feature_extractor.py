import torch

from torch import nn

class ConvolutionalFeatureExtractor(nn.Module):

    def __init__(self, num_seismic_components, final_feature_length, 
                     should_concat_location : bool) -> None:
        super().__init__()

        self.seismic_trace_CNN = SeismicTraceCNN(num_seismic_components)
        self.feature_combination_operation = ConcatLayer() if should_concat_location else None ### TODO: Implement sinusoidal embeddings option 

        self.feedforward_net = FeedForwardFeatureProcessing((100,100,50), 
                                    output_dim = final_feature_length)


    def forward(self, x_trace):

        cnn_processed_trace = self.seismic_trace_CNN.forward(x_trace)
        # cnn_plus_location = self.feature_combination_operation([cnn_processed_trace, x_location])

        return self.feedforward_net(cnn_processed_trace)


class ConcatLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, iterable_to_be_concatenated):
        return torch.cat(iterable_to_be_concatenated, dim = 1)


class SeismicTraceCNN(nn.Module):

    def __init__(self, num_seismic_components) -> None:
        super().__init__()

        self.flatten = nn.Flatten()

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(num_seismic_components, 64, 3, 2),
                nn.ReLU(),
                nn.Conv1d(64, 64, 3, 2),
                nn.ReLU(),
                nn.Conv1d(64, 64, 3, 2),
                nn.ReLU(),
                nn.Conv1d(64, 64, 3, 2),
                nn.ReLU(),
                nn.Conv1d(64, 16, 3, 1),
                nn.ReLU(),
                nn.Conv1d(16, 4, 3, 1),
                nn.ReLU(),
            ]
        )

    def forward(self, x):

        max_trace_val = x.abs().amax(dim=(1,2), keepdim=True)
        scaled_x = x / max_trace_val

        for conv in self.convs:
            scaled_x = conv.forward(scaled_x)

        scaled_x = self.flatten(scaled_x)
        scaled_x = torch.cat([scaled_x, max_trace_val.log()[:,0]], dim = 1)

        return scaled_x


class FeedForwardFeatureProcessing(nn.Module):

    def __init__(self, hidden_layer_dims, output_dim) -> None:
        super().__init__()
        
        self.ffw_hidden_layers = nn.ModuleList([nn.Sequential(*[nn.LazyLinear(num_nodes), nn.ReLU()]) for num_nodes in hidden_layer_dims])
        self.last_layer = nn.LazyLinear(output_dim)

    def forward(self, x):

        for ffw_layer in self.ffw_hidden_layers:
            x = ffw_layer.forward(x)

        return self.last_layer(x)