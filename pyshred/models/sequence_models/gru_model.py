import torch.nn as nn
import torch
from .abstract_sequence import AbstractSequence
from ..decoder_models.mlp_model import MLP
from ..decoder_models.unet_model import UNET

class GRU(AbstractSequence):
    """
    GRU (Gated Recurrent Unit) sequence model for encoding temporal sensor dynamics.
    
    A recurrent neural network that processes sensor measurement sequences
    to learn latent representations of the underlying dynamics. Supports
    optional layer normalization for improved training stability.

    Parameters
    ----------
    hidden_size : int, optional
        Size of the GRU hidden state. Defaults to 3.
    num_layers : int, optional
        Number of GRU layers. Defaults to 1.
    layer_norm : bool, optional
        Whether to apply layer normalization. Defaults to False.

    Attributes
    ----------
    hidden_size : int
        Size of the GRU hidden state.
    num_layers : int
        Number of GRU layers.
    use_layer_norm : bool
        Whether layer normalization is applied.
    output_size : int
        Size of the output (equals hidden_size).
    """

    def __init__(self, hidden_size: int = 3, num_layers: int = 1, layer_norm: bool = False):
        """
        Initialize the GRU sequence model.

        Parameters
        ----------
        hidden_size : int, optional
            Size of the GRU hidden state. Defaults to 3.
        num_layers : int, optional
            Number of GRU layers. Defaults to 1.
        layer_norm : bool, optional
            Whether to apply layer normalization. Defaults to False.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = None  # lazy initialization
        self.decoder_type = None
        self.output_size = hidden_size
        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)

    def initialize(self, input_size: int, decoder_type, **kwargs):
        """
        Initialize the GRU with input size and decoder.

        Parameters
        ----------
        input_size : int
            Number of input features (sensor measurements).
        decoder_type : str
            Decoder model type.
        **kwargs
            Additional keyword arguments.
        """
        super().initialize(input_size)
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder_type = decoder_type

    def forward(self, x):
        """
        Forward pass through the GRU model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor with latent representations. Shape depends on decoder type:
            - MLP decoder: (batch_size, hidden_size)
            - UNET decoder: (batch_size, hidden_size, sequence_length)

        Raises
        ------
        TypeError
            If decoder type is not supported.
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

        if self.decoder_type == "MLP":
            return h_last.view(-1, self.hidden_size)
        elif self.decoder_type == "UNET":
            return out.permute(0, 2, 1)
        else:
            raise TypeError(
                f"Unsupported decoder type: {self.decoder_type}."
            )

    @property
    def model_name(self):
        """
        Name of the sequence model.

        Returns
        -------
        str
            Returns "GRU".
        """
        return "GRU"
