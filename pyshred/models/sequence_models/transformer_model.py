import torch
import math
import torch.nn as nn
from .abstract_sequence import AbstractSequence
from ..decoder_models.mlp_model import MLP
from ..decoder_models.unet_model import UNET

class TRANSFORMER(AbstractSequence):
    """
    Transformer-based sequence model for encoding temporal sensor dynamics.
    
    Uses self-attention mechanisms to capture long-range dependencies in
    sensor measurement sequences. Includes positional encoding and supports
    optional layer normalization.

    Parameters
    ----------
    d_model : int, optional
        Dimensionality of the model (embedding size). Defaults to 128.
    nhead : int, optional
        Number of attention heads. Defaults to 16.
    dropout : float, optional
        Dropout probability. Defaults to 0.2.
    layer_norm : bool, optional
        Whether to apply layer normalization. Defaults to False.

    Attributes
    ----------
    d_model : int
        Model dimensionality.
    hidden_size : int
        Hidden size (same as d_model).
    output_size : int
        Output size (same as d_model).
    use_layer_norm : bool
        Whether layer normalization is applied.
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 16, dropout: float = 0.2, layer_norm: bool = False):
        """
        Initialize the Transformer sequence model.

        Parameters
        ----------
        d_model : int, optional
            Dimensionality of the model (embedding size). Defaults to 128.
        nhead : int, optional
            Number of attention heads. Defaults to 16.
        dropout : float, optional
            Dropout probability. Defaults to 0.2.
        layer_norm : bool, optional
            Whether to apply layer normalization. Defaults to False.
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model
        self.dropout = dropout
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, activation=nn.GELU(), batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.relu = nn.ReLU()
        self.output_size = d_model
        self.decoder = None
        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_size)

    def initialize(self, input_size:int, lags:int, decoder_type, **kwargs):
        """
        Initialize the Transformer with input size, sequence length, and decoder.

        Parameters
        ----------
        input_size : int
            Number of input features (sensor measurements).
        lags : int
            Length of input sequences.
        decoder_type : str
            Decoder model type.
        **kwargs
            Additional keyword arguments.
        """
        super().initialize(input_size)
        sequence_length = lags
        self.pos_encoder = PositionalEncoding(self.d_model,
                                              sequence_length,
                                              self.dropout)
        self.input_embedding = nn.GRU(input_size=self.input_size
                                      ,hidden_size=self.d_model,
                                      num_layers=2,
                                      batch_first=True)
        self.decoder_type = decoder_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor with latent representations. Shape depends on decoder type:
            - MLP decoder: (batch_size, d_model)
            - UNET decoder: (batch_size, d_model, sequence_length)

        Raises
        ------
        TypeError
            If decoder type is not supported.
        """
        super().forward(x)
        # Apply input embedding
        x, _= self.input_embedding(x)
        # Apply positional encoding
        x = self.pos_encoder(x)
        # Apply transformer encoder
        x = self.transformer_encoder(x, self._generate_square_subsequent_mask(x.size(1), x.device).to(torch.bool))

        # Apply layer normalization if enabled
        if self.use_layer_norm:
            x = self.layer_norm(x)

        if self.decoder_type == "MLP":
            return x[:,-1,:]
        elif self.decoder_type == "UNET":
            return x.permute(0, 2, 1)
        else:
            raise TypeError(
                f"Unsupported decoder type: {self.decoder_type}. "
            )

    def _generate_square_subsequent_mask(self, sequence_length: int, device) -> torch.Tensor:
        """
        Generate a causal mask for the transformer attention.

        Parameters
        ----------
        sequence_length : int
            Length of the sequence.
        device : torch.device
            Device to create the mask on.

        Returns
        -------
        torch.Tensor
            Boolean mask tensor of shape (sequence_length, sequence_length).
        """
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1).bool()
        return mask

    @property
    def model_name(self):
        """
        Name of the sequence model.

        Returns
        -------
        str
            Returns "Transformer".
        """
        return "Transformer"

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    
    Adds learnable positional information to input embeddings to help
    the model understand sequence order.

    Parameters
    ----------
    d_model : int
        Dimensionality of the model embeddings.
    max_sequence_length : int, optional
        Maximum sequence length to precompute encodings for. Defaults to 5000.
    dropout : float, optional
        Dropout probability. Defaults to 0.1.

    Attributes
    ----------
    pe : torch.Tensor
        Precomputed positional encoding tensor.
    """
    
    def __init__(self, d_model, max_sequence_length=5000, dropout=0.1):
        """
        Initialize positional encoding.

        Parameters
        ----------
        d_model : int
            Dimensionality of the model embeddings.
        max_sequence_length : int, optional
            Maximum sequence length to precompute encodings for. Defaults to 5000.
        dropout : float, optional
            Dropout probability. Defaults to 0.1.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute the positional encodings for a maximum sequence length
        position = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_sequence_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        pos_encoding = pos_encoding.unsqueeze(0)  # Shape: (1, max_sequence_length, d_model)
        self.register_buffer('pe', pos_encoding)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Input with positional encoding added, same shape as input.
        """
        # x.shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]  # Adjust to the sequence length of the input
        x = x + pe
        return self.dropout(x)