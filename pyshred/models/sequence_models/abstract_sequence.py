from abc import ABC, abstractmethod
import torch.nn as nn


class AbstractSequence(ABC, nn.Module):
    """
    Abstract base class for all sequence models.
    """

    def __init__(self):
        """
        Lazily initialize the sequence model because
        `input_size`is typically provided after model initialization.
        """
        super().__init__() # initialize nn.Module
        self.is_initialized = False # lazy initialization flag

    @abstractmethod
    def initialize(self, input_size):
        self.input_size = input_size
        self.is_initialized = True

    @abstractmethod
    def forward(self, x):
        """
        Forward pass through the sequence model.
        """
        if not self.is_initialized:
            raise RuntimeError("The sequence model is not initialized. Call `initialize` first.")
        pass

    @property
    @abstractmethod
    def model_name(self):
        """
        Returns the name of the sequence model.

        Returns:
        --------
        str
            The name of the model.
        """
        pass