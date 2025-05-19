import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMDynamics(nn.Module):
    """
    LSTM‐based latent‐space dynamics:
    learns to forecast the next latent vector given the current one.
    """

    def __init__(self,
                 latent_dim: int,
                 hidden_dim: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        # We'll feed each z_t as a one‐step sequence (batch, 1, latent_dim)
        self.lstm = nn.LSTM(
            input_size = latent_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )
        # Project LSTM hidden output back to latent_dim
        self.project = nn.Linear(hidden_dim, latent_dim)

    def predict_next(self,
                     z: torch.Tensor,   # shape (batch, latent_dim)
                     dt: float = None   # unused, but kept for API parity
    ) -> torch.Tensor:
        # prepare as sequence length=1
        z_seq = z.unsqueeze(1)                      # (batch, 1, latent_dim)
        out, (h_n, _) = self.lstm(z_seq)            # out: (batch, 1, hidden_dim)
        h_last = out[:, -1, :]                      # (batch, hidden_dim)
        return self.project(h_last)                 # (batch, latent_dim)

    def loss(self,
             z_curr: torch.Tensor,  # (batch, latent_dim)
             z_next: torch.Tensor   # (batch, latent_dim)
    ) -> torch.Tensor:
        z_pred = self.predict_next(z_curr)
        return F.mse_loss(z_pred, z_next)

    def simulate(self,
                 z0: torch.Tensor,     # shape (latent_dim,) or (1, latent_dim)
                 timesteps: int
    ) -> torch.Tensor:
        # roll forward one step at a time
        zs = [z0.view(1, -1)]                       # list of (1, latent_dim)
        h, c = None, None
        for _ in range(timesteps):
            # each step we run the LSTMCell directly or use predict_next:
            z_prev = zs[-1]                         # (1, latent_dim)
            # if you want to carry hidden state: unpack LSTMCell here
            z_next = self.predict_next(z_prev)
            zs.append(z_next)
        return torch.cat(zs, dim=0)                # (timesteps+1, latent_dim)
