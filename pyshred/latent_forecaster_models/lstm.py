import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    """
    Forecaster that takes recent latent-space embeddings and predicts
    the next sensor measurement.
    """
    def __init__(
        self,
        latent_dim:    int,
        hidden_dim:    int,
        num_layers:    int,
        output_dim:    int,   # number of sensors
        dropout:       float  = 0.0,
        bidirectional: bool   = False,
    ):
        super().__init__()
        self.latent_dim    = latent_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM to forecast the next latent
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers>1 else 0.0,
            bidirectional=bidirectional
        )
        # Map from LSTM hidden to next latent
        self.fc_latent = nn.Linear(hidden_dim * self.num_directions, latent_dim)
        # Finally map that latent to sensor space
        self.fc_sensor = nn.Linear(latent_dim, output_dim)


    # NEED TO GENERATE LAGGED LATENTS
    def forward(self, lagged_latents: torch.Tensor) -> torch.Tensor:
        """
        lagged_latents: (batch, seq_len, latent_dim)
        returns: next_sensor: (batch, output_dim)
        """
        batch, seq_len, _ = lagged_latents.shape

        # 1) forecast next latent
        lstm_out, _ = self.lstm(lagged_latents)  
        # take last time step
        h_last = lstm_out[:, -1, :]                # (batch, hidden_dim * directions)
        z_next = self.fc_latent(h_last)            # (batch, latent_dim)

        # 2) decode to sensor space
        sensor_next = self.fc_sensor(z_next)       # (batch, output_dim)
        return sensor_next

    def fit(latents):
        pass
