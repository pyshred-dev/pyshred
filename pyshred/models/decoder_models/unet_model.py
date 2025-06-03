import torch
import torch.nn as nn
from .abstract_decoder import AbstractDecoder
import torch.nn as nn


class UNET(AbstractDecoder):
    """
    1D U-Net style convolutional decoder for SHRED.
    
    Uses a series of 1D convolutions to decode latent representations
    back to the physical state space. Particularly suitable for 
    spatially-structured data.

    Parameters
    ----------
    conv1 : int, optional
        Number of channels in first convolutional layer. Defaults to 256.
    conv2 : int, optional
        Number of channels in second convolutional layer. Defaults to 1024.

    Attributes
    ----------
    c1 : int
        First convolution layer channel count.
    c2 : int
        Second convolution layer channel count.
    """
    
    def __init__(self, conv1: int = 256, conv2: int = 1024):
        """
        Initialize the UNET decoder.

        Parameters
        ----------
        conv1 : int, optional
            Number of channels in first convolutional layer. Defaults to 256.
        conv2 : int, optional
            Number of channels in second convolutional layer. Defaults to 1024.
        """
        super().__init__()
        self.c1 = conv1
        self.c2 = conv2
        # self.dropout = dropout


    def initialize(self, input_size, output_size):
        """
        Initialize the UNET decoder with input and output sizes.

        Parameters
        ----------
        input_size : int
            Size of the input latent features.
        output_size : int
            Size of the output physical state.
        """
        super().initialize(input_size)
        self.conv1 = nn.Conv1d(input_size, self.c1, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(self.c1, self.c2, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(self.c2, output_size, kernel_size=2, padding=1)
        self.gelu = nn.LeakyReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the UNET decoder.

        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor of shape (batch_size, input_size, sequence_length).

        Returns
        -------
        torch.Tensor
            Decoded output tensor of shape (batch_size, output_size).
        """
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        x = x.permute(0, 2, 1)  # Change shape back to [batch_size, sequence_length, d_model]
        x = torch.mean(x, dim=1)
        return x

    @property
    def model_name(self):
        """
        Name of the decoder model.

        Returns
        -------
        str
            Returns "UNET".
        """
        return "UNET"