from abc import ABC, abstractmethod
import torch.nn as nn

class AbstractLatentForecaster(nn.Module, ABC):
    """
    Abstract base class for latent space forecasters.
    All latent forecasters must implement `initialize`, `fit`, and `forecast`.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def initialize(self, latent_dim):
        """
        Initialize internal structures based on latent dimensionality.
        Args:
            latent_dim (int): Dimension of the latent space.
        """
        pass

    @abstractmethod
    def fit(self, latents, *args, **kwargs):
        """
        Train the forecaster on latent space data.
        Args:
            latents: Latent time series data (e.g., shape (T, D)).
        """
        pass

    @abstractmethod
    def forecast(self, h, init_latents):
        """
        Forecast the next `h` latent steps based on initial latents.
        Args:
            h (int): Number of steps to forecast.
            init_latents: Tensor or array of shape (>=lags, D) or (D,)
        Returns:
            Forecasted latent states.
        """
        pass
