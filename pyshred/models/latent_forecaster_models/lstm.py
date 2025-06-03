import warnings
import torch
import torch.nn as nn
import numpy as np
from .abstract_latent_forecaster import AbstractLatentForecaster

class LSTM_Forecaster(AbstractLatentForecaster):
    """
    LSTM-based forecaster for latent space dynamics.

    Parameters
    ----------
    hidden_size : int, optional
        Size of LSTM hidden state. Defaults to 64.
    num_layers : int, optional
        Number of LSTM layers. Defaults to 1.
    lags : int, optional
        Number of time lags to use for forecasting. Defaults to 20.

    Attributes
    ----------
    seed_length : int
        Number of latent space timesteps required to seed forecaster.
    latent_dim : int
        Dimension of the latent space.
    """

    def __init__(self, hidden_size=64, num_layers=1, lags=20):
        super().__init__()
        self.seed_length = lags  # number of latent space timesteps required to seed forecaster
        self.lags = lags
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    # input_size = latent_dim
    def initialize(self, latent_dim):
        """
        Initialize the LSTM with the given latent dimension.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        """
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(latent_dim, self.hidden_size, self.num_layers,
                    batch_first=True)
        self.proj = nn.Linear(self.hidden_size, latent_dim)


    def forward(self, x):
        """
        Forward pass through the LSTM forecaster.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, lags, latent_dim).

        Returns
        -------
        torch.Tensor
            Forecasted latent state of shape (batch, latent_dim).
        """
        # x: (batch, lags, latent_dim)
        out, _ = self.lstm(x)        # out: (batch, lags, hidden_size)
        last = out[:, -1, :]         # (batch, hidden_size)
        return self.proj(last)       # (batch, latent_dim)

    def fit(self, latents, num_epochs, batch_size, lr):
        """
        data: torch.Tensor of shape (T, D)
        """
        device = next(self.parameters()).device
        data = torch.tensor(latents, dtype=torch.float32, device=device)
        T, D = data.shape
        inputs, targets = [], []
        for i in range(T - self.lags):
            inputs.append(data[i:i+self.lags])
            targets.append(data[i+self.lags])
        X = torch.stack(inputs)       # (N, lags, D)
        Y = torch.stack(targets)      # (N, D)

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader  = torch.utils.data.DataLoader(dataset,
                   batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.train()
        for ep in range(num_epochs):
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = self(xb)
                loss = loss_fn(pred, yb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def forecast(self, h, init_latents):
        """
        init_latents: torch.Tensor of shape (>=lags, D)
        returns: Tensor of shape (h, D)
        """
        self.eval()
        if init_latents.ndim != 2:
            raise ValueError(
                f"Invalid `init_latents`: expected a 2D array (shape (timesteps, m)), "
                f"but got a {init_latents.ndim}D array with shape {init_latents.shape}."
            )
        total_steps, D = init_latents.shape
        if total_steps < self.lags:
            raise ValueError(f"Expected init_latents to have at least {self.lags} steps, "
                            f"but got {total_steps}")
        if total_steps > self.lags:
            warnings.warn(
                f"`init_latents` has shape {init_latents.shape}; only the last {self.lags} rows "
                "will be used as the initial latent state for SINDy.",
                UserWarning
            )
        device = next(self.parameters()).device
        # only keep the last `lags` steps
        seed = torch.tensor(init_latents[-self.lags :], dtype=torch.float32, device=device)
        window = seed .clone().unsqueeze(0)  # (1, lags, D)
        preds = []
        for _ in range(h):
            p = self(window)                 # (1, D)
            preds.append(p.squeeze(0))
            # roll the window: drop oldest, append p
            window = torch.cat([window[:,1:,:], p.unsqueeze(1)], dim=1)
        result = torch.stack(preds)                    # (h, D) tensor on device
        return result.cpu().numpy()                   # (h, D) ndarray on CPU