import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .sindy_dynamics import SINDyDynamics
from .decoder_models import *
from .sequence_models import *
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np


SEQUENCE_MODELS = {
    "LSTM": LSTM,
    "TRANSFORMER": TRANSFORMER,
    "GRU": GRU,
}

DECODER_MODELS = {
    "SDN": SDN,
    "UNET": UNET,
}

class SHRED(nn.Module):
    """
    Shallow Recurrent Decoder with optional SINDy dynamics regularization.
    """
    def __init__(self, sequence_model = None, decoder_model = None, dynamics = None, lambda_dyn = 1e-2):
        super().__init__()
        self.dynamics = dynamics
        self.lambda_dyn = lambda_dyn
        if sequence_model is None:
            if dynamics is not None:
                self.sequence = GRU()
            else:
                self.sequence = LSTM()
        if decoder_model is None:
            self.decoder = SDN()

    def forward(self, x: torch.Tensor, dt: Optional[float] = None, sindy: bool = False):
        # x: (batch, lags, sensors)
        z = self.sequence(x)        # (batch, latent_dim)
        x_rec = self.decoder(z)     # (batch, high_dim)
        if sindy and self.dynamics:
            # for dynamics, interpret batch as time sequence
            z = z['final_hidden_state']
            z_curr = z[:-1] # all but last time
            z_next = z[1:] # all but first time
            z_pred = self.dynamics.predict_next(z_curr, dt or self.dynamics.dt)
            return x_rec, z_curr, z_next, z_pred
        return x_rec

    def fit(self, train_dataset, val_dataset,  batch_size=64, num_epochs=200, lr=1e-3, sindy=True, verbose=True, patience=20):

        input_size = train_dataset.X.shape[2] # nsensors + nparams
        output_size = train_dataset.Y.shape[1]
        lags = train_dataset.X.shape[1] # lags

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence.initialize(input_size=input_size, lags=lags)
        self.sequence.to(device)
        self.decoder.initialize(input_size=self.sequence.output_size, output_size=output_size)
        self.decoder.to(device)
        self.to(device)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        val_error_list = []
        patience_counter = 0
        best_params = {k: v.cpu().clone() for k, v in self.state_dict().items()}
        best_val_error = float('inf')  # Initialize with a large value

        for epoch in range(1, num_epochs + 1):
            self.train()
            running_loss = 0.0
            running_l2 = 0.0
            if verbose:
                pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}', unit='batch')
            for X, Y in train_loader:
                X, Y = X.to(device), Y.to(device)
                optimizer.zero_grad()

                # forward + loss
                if sindy and hasattr(self, "dynamics") and self.dynamics:
                    x_rec, z_curr, z_next, _ = self(X, dt=self.dynamics.dt, sindy=True)
                    loss_rec = criterion(x_rec, Y)
                    loss_dyn = self.dynamics.loss(z_curr, z_next)
                    loss = loss_rec + self.lambda_dyn * loss_dyn
                else:
                    x_rec = self(X)
                    loss = criterion(x_rec, Y)

                loss.backward()
                optimizer.step()
                # accumulate metrics
                running_loss += loss.item()
                l2_err = torch.norm(Y - x_rec).item()
                running_l2  += l2_err

                if verbose:
                    pbar.set_postfix({
                        "loss": running_loss / (pbar.n + 1),
                        "L2":   running_l2  / (pbar.n + 1),
                    })
                    pbar.update(1)

            if verbose:
                pbar.close()

            self.eval()
            with torch.no_grad():
                X_val, Y_val = val_dataset.X.to(device), val_dataset.Y.to(device)
                if sindy and hasattr(self, "dynamics") and self.dynamics:
                    x_rec_val, _, _, _ = self(X_val, dt=self.dynamics.dt, sindy=True)
                else:
                    x_rec_val = self(X_val)

                val_l2 = torch.norm(Y_val - x_rec_val).item()
                val_error_list.append(val_l2)

            if verbose:
                print(
                    f"Epoch {epoch}: "
                    f"train_loss={running_loss/len(train_loader):.4f}, "
                    f"train_L2={running_l2/len(train_loader):.4f}, "
                    f"val_L2={val_l2:.4f}"
                )
            # --- 5) early stopping check ---
            if val_l2 < best_val_error:
                best_val_error = val_l2
                best_params = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                self._best_L2_error = val_l2
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print("Early stopping: no val improvement for "
                            f"{patience} epochs.")
                    break

        # --- 6) restore best model & return val errors ---
        # load back onto the correct device
        restored = {k: v.to(device) for k, v in best_params.items()}
        self.load_state_dict(restored)

        return np.array(val_error_list)