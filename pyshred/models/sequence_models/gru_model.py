import torch.nn as nn
import torch
from .abstract_sequence import AbstractSequence
from ..decoder_models.mlp_model import MLP
from ..decoder_models.unet_model import UNET

class GRU(AbstractSequence):

    def __init__(self, hidden_size: int = 3, num_layers: int = 1, layer_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None  # lazy initialization
        self.decoder = None
        self.output_size = hidden_size
        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)

    def initialize(self, input_size: int, decoder, **kwargs):
        super().initialize(input_size)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = decoder

    def forward(self, x):
        """
        Forward pass through the GRU model.
        """
        super().forward(x)
        device = next(self.parameters()).device
        # Initialize hidden state
        h_0 = torch.zeros((self.num_layers, x.size(0), self.hidden_size), device=device)
        out, h_out = self.gru(x, h_0)

        # Apply layer normalization if enabled
        if self.use_layer_norm:
            # Normalize per time-step across hidden dimension
            out = self.layer_norm(out)
            # Normalize the final hidden state from last layer
            h_last = self.layer_norm(h_out[-1])
        else:
            h_last = h_out[-1]

        if isinstance(self.decoder, MLP):
            return h_last.view(-1, self.hidden_size)
        elif isinstance(self.decoder, UNET):
            return out.permute(0, 2, 1)
        else:
            raise TypeError(
                f"Unsupported decoder type: {type(self.decoder).__name__}."
            )

    @property
    def model_name(self):
        return "GRU"
