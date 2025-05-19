from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractDecoder(ABC, nn.Module):
    """
    Abstract base class for all decoder models.
    """


    def __init__(self):
        """
        Lazily initialize the decoder model because
        `input_size`is typically provided after model initialization.
        """
        super().__init__() # initialize nn.Module
        self.is_initialized = False # lazy initialization flag


    @abstractmethod
    def initialize(self, input_size):
        """
        Initialize the decoder model with input and output sizes.

        Parameters:
        -----------
        input_size : int
            Size of the input features.
        output_size : int
            Size of the output features.
        """
        self.input_size = input_size
        self.is_initialized = True

    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the decoder model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, output_size).
        """
        if not self.is_initialized:
            raise RuntimeError("The decoder model is not initialized. Call `initialize` first.")
        pass

    @property
    @abstractmethod
    def model_name(self):
        """
        Returns the name of the decoder model.

        Returns:
        --------
        str
            The name of the model.
        """
        pass