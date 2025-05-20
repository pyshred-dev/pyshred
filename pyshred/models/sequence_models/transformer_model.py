import torch
import math
import torch.nn as nn
from .abstract_sequence import AbstractSequence
from ..decoder_models.sdn_model import SDN
from ..decoder_models.unet_model import UNET

class TRANSFORMER(AbstractSequence):
    def __init__(self, d_model: int = 128, nhead: int = 16, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model
        self.dropout = dropout
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, activation=nn.GELU(), batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.relu = nn.ReLU()
        self.output_size = d_model
        self.decoder = None

    def initialize(self, input_size:int, lags:int, decoder, **kwargs):
        super().initialize(input_size)
        sequence_length = lags
        self.pos_encoder = PositionalEncoding(self.d_model,
                                              sequence_length,
                                              self.dropout)
        self.input_embedding = nn.GRU(input_size=self.input_size
                                      ,hidden_size=self.d_model,
                                      num_layers=2,
                                      batch_first=True)
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        super().forward(x)
        # Apply input embedding
        x, _= self.input_embedding(x)
        # Apply positional encoding
        x = self.pos_encoder(x)
        # Apply transformer encoder
        x = self.transformer_encoder(x, self._generate_square_subsequent_mask(x.size(1), x.device).to(torch.bool))
        if isinstance(self.decoder, SDN):
            return x[:,-1,:]
        elif isinstance(self.decoder, UNET):
            return x.permute(0, 2, 1)
        else:
            raise TypeError(
                f"Unsupported decoder type: {type(self.decoder).__name__}. "
            )

    def _generate_square_subsequent_mask(self, sequence_length: int, device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sequence_length, sequence_length, device=device), diagonal=1).bool()
        return mask


    @property
    def model_name(self):
        return "Transformer"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length=5000, dropout=0.1):
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
        # x.shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]  # Adjust to the sequence length of the input
        x = x + pe
        return self.dropout(x)