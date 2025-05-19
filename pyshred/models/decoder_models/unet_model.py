import torch
import torch.nn as nn
from .abstract_decoder import AbstractDecoder
import torch.nn as nn


class UNET(AbstractDecoder):
    def __init__(self, dropout: float = 0.1, conv1: int = 256, conv2: int = 1024):
        super().__init__()
        self.c1 = conv1
        self.c2 = conv2
        # self.dropout = dropout


    def initialize(self, input_size, output_size):
        """
        Initialize the SDNDecoder with input and output sizes.

        Parameters:
        -----------
        input_size : int
            Size of the input features.
        output_size : int
            Size of the output features.
        """
        super().initialize(input_size)
        # self.dropoutLayer = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv1d(input_size, self.c1, kernel_size=2, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv1
        self.conv2 = nn.Conv1d(self.c1, self.c2, kernel_size=4, padding=1)
        # self.conv_transpose1 = nn.ConvTranspose1d(in_channels=512, out_channels=1024, kernel_size=2, padding=1)
        # self.conv_transpose2 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv2
        self.conv3 = nn.Conv1d(self.c2, output_size, kernel_size=2, padding=1)
        # self.pool3 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer after conv3
        # self.conv4 = nn.Conv1d(283, 128, kernel_size=2, padding=1)
        # self.conv5 = nn.Conv1d(128, 283, kernel_size=4, padding=1)
        # self.relu = nn.ReLU()
        self.gelu = nn.LeakyReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x["sequence_output"]
        # Assuming x has shape [batch_size, sequence_length, d_model]
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, d_model, sequence_length]
        # print('x.shape', x.shape)
        # Pass through the Conv1d, BatchNorm, GELU, and MaxPool layers
        x = self.gelu(self.conv1(x))
        x = self.gelu(self.conv2(x))
        x = self.gelu(self.conv3(x))
        # print(x.shape)
        # x = self.gelu(self.b10(self.conv10(x)))
        # x = self.pool10(x)
        # x = self.gelu(self.b11(self.conv11(x)))
        # x = self.pool11(x)
        # Optionally, permute back to the original shape if needed
        x = x.permute(0, 2, 1)  # Change shape back to [batch_size, sequence_length, d_model]
        x = torch.mean(x, dim=1)
        return x


    def model_name(self):
        return "UNET"