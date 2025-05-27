import torch.nn as nn
import torch
from .abstract_sequence import AbstractSequence
from ..decoder_models.mlp_model import MLP
from ..decoder_models.unet_model import UNET

class GRU(AbstractSequence):

    def __init__(self, hidden_size: int = 3, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None  # lazy initialization
        self.decoder = None
        self.output_size = hidden_size

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
        if isinstance(self.decoder, MLP):
            return h_out[-1].view(-1, self.hidden_size)
        elif isinstance(self.decoder, UNET):
            return out.permute(0, 2, 1)
        else:
            raise TypeError(
                f"Unsupported decoder type: {type(self.decoder).__name__}. "
            )

    @property
    def model_name(self):
        return "GRU"
