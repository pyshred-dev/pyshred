import torch
from torch.utils.data import DataLoader
import numpy as np

from ..latent_forecaster_models.lstm import LSTMForecaster
from .sindy_dynamics import SINDyDynamics
from ..latent_forecaster_models.sindy import SINDy_Forecaster
import pysindy as ps
from .sindy import sindy_library_torch, e_sindy_library_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SINDy(torch.nn.Module):
    def __init__(self, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h, dt):
        library_Theta = sindy_library_torch(h, self.latent_dim, self.poly_order, self.include_sine)
        h = h + library_Theta @ (self.coefficients * self.coefficient_mask) * dt
        return h
    
    def thresholding(self, threshold):
        self.coefficient_mask = torch.abs(self.coefficients) > threshold
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)   

class E_SINDy(torch.nn.Module):
    def __init__(self, num_replicates, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h_replicates, dt):
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(h_replicates, self.latent_dim, self.poly_order, self.include_sine)
        library_Thetas = library_Thetas.reshape(num_data, num_replicates, self.library_dim)
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = h_replicates + torch.einsum('ijk,jkl->ijl', library_Thetas, (self.coefficients * self.coefficient_mask)) * dt
        return h_replicates
    
    def thresholding(self, threshold, base_threshold=0):
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)   
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)  

from .decoder_models import *
from .sequence_models import *

class SINDy_SHRED(torch.nn.Module):
    def __init__(self, sequence_model = None, decoder_model = None, dynamics = None, layer_norm = False):
        super(SINDy_SHRED, self).__init__()
        self.dynamics = dynamics
        if sequence_model is None:
            if dynamics is not None:
                self.sequence = GRU()
            else:
                self.sequence = LSTM()
        if decoder_model is None:
            self.decoder = SDN()
        self.num_replicates = 10
        self.e_sindy = E_SINDy(self.num_replicates, self.sequence.hidden_size, dynamics.lib_dim, dynamics.poly_order,
                               dynamics.include_sine).to(device)
        self.dt = dynamics.dt
        self.poly_order = dynamics.poly_order
        self.use_layer_norm = layer_norm
        self.layer_norm_gru = torch.nn.LayerNorm(self.sequence.hidden_size)
        self.latent_forecaster = None

    def forward(self, x, sindy=False):
        if sindy == True:
            h_out = self.sequence(x)
            h_out = h_out["final_hidden_state"]
            if self.use_layer_norm:
                h_out = self.layer_norm_gru(h_out)
            decoder_input = {"final_hidden_state": h_out}
            output = self.decoder(decoder_input)
            with torch.autograd.set_detect_anomaly(True):
                if sindy:
                    h_t = h_out[:-1, :]
                    ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
                    for _ in range(10):
                        ht_replicates = self.e_sindy(ht_replicates, dt=self.dt)
                    h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
                    output = output, h_out_replicates, ht_replicates
        else:
            h_out = self.sequence(x)
            output = self.decoder(h_out)
        return output

    
    def gru_outputs(self, x, sindy=False):
        if sindy == True:
            h_out = self.sequence(x)
            h_out = h_out["final_hidden_state"]
            if self.use_layer_norm:
                h_out = self.layer_norm_gru(h_out)
            if sindy:
                h_t = h_out[:-1, :]
                ht_replicates = h_t.unsqueeze(1).repeat(1, self.num_replicates, 1)
                for _ in range(10):
                    ht_replicates = self.e_sindy(ht_replicates, dt=self.dt)
                h_out_replicates = h_out[1:, :].unsqueeze(1).repeat(1, self.num_replicates, 1)
                h_outs = h_out_replicates, ht_replicates
        else:
            h_outs = self.sequence(x)
        return h_outs

    def sindys_threshold(self, threshold):
        self.e_sindy.thresholding(threshold)

    def sindys_add_noise(self, noise):
        self.e_sindy.add_noise(noise)

    # sindy_regularization previously set to 1.0
    def fit(self, train_dataset, val_dataset,  batch_size=64, num_epochs=200, lr=1e-3, sindy_regularization=0, optimizer="AdamW", verbose=True, threshold=0.5, base_threshold=0.0, patience=20, thres_epoch=100, weight_decay=0.01):
        if sindy_regularization > 0:
            sindy = True # sindy regularization is on
        else:
            sindy = False
        input_size = train_dataset.X.shape[2] # nsensors + nparams
        output_size = train_dataset.Y.shape[1]
        lags = train_dataset.X.shape[1] # lags
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence.initialize(input_size=input_size, lags=lags)
        self.sequence.to(device)
        self.decoder.initialize(input_size=self.sequence.output_size, output_size=output_size)
        self.decoder.to(device)
        self.to(device)
        # shuffle is False (used to be True)
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size)
        criterion = torch.nn.MSELoss()
        if optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        val_error_list = []
        patience_counter = 0
        best_params = self.state_dict()
        best_val_error = float('inf')  # Initialize with a large value
        if sindy == True:
            print("Fitting SindySHRED...")
        else:
            print("Fitting SHRED...")
        for epoch in range(1, num_epochs + 1):
            self.train()
            running_loss = 0.0
            for data in train_loader:
                if sindy:
                    outputs, h_gru, h_sindy = self(data[0], sindy=True)
                    optimizer.zero_grad()
                    loss = criterion(outputs, data[1]) + criterion(h_gru, h_sindy) * sindy_regularization + torch.abs(torch.mean(h_gru)) * 0.1
                else:
                    outputs = self(data[0], sindy=False)
                    optimizer.zero_grad()
                    loss = criterion(outputs, data[1])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            if verbose:
                print(f"Epoch {epoch}: Average training loss = {running_loss / len(train_loader):.6f}")
            if sindy:
                if epoch % thres_epoch == 0 and epoch != 0:
                    self.e_sindy.thresholding(threshold=threshold, base_threshold=base_threshold)
            self.eval()
            with torch.no_grad():
                val_outputs = self(val_dataset.X.to(device))
                val_targets = val_dataset.Y.to(device)
                val_error = criterion(val_outputs, val_targets)
                val_error_list.append(val_error)
            if verbose:
                print('Training epoch ' + str(epoch))
                print('Error ' + str(val_error_list[-1]))
            if val_error < best_val_error:
                best_val_error = val_error
                best_params = self.state_dict()
                patience_counter = 0  # Reset if improvement
            else:
                patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print("Early stopping triggered: patience threshold reached.")
                break
        self.load_state_dict(best_params)
        device = next(self.parameters()).device
        X_train = train_dataset.X.to(device)    # shape (N_train, lags, n_sensors)
        X_val   = val_dataset.X.to(device)      # shape (N_val,   lags, n_sensors)
        X_all = torch.cat([X_train, X_val], dim=0)   # shape (N_train+N_val, lags, n_sensors)
        # run through encoder to get latents
        self.eval()
        with torch.no_grad():
            latents = self.gru_outputs(X_all, sindy=False)   # (N_train+N_val, latent_dim)
            latents = latents["final_hidden_state"]
        print('X_all.shape',X_all.shape)
        print('latents.shape')
        # to numpy and hand off to pysindy
        latents_np = latents.cpu().numpy()
        if isinstance(self.dynamics, SINDyDynamics):
            self.latent_forecaster = SINDy_Forecaster(
                latents_np,
                self.dt,
                poly_order=self.poly_order,
                optimizer=ps.STLSQ(threshold=0.0, alpha=0.05),
                diff_method=ps.differentiation.FiniteDifference()
            )
        # if isinstance(self.dynamics, LSTMDynamics):
        #     pass
        return torch.tensor(val_error_list).detach().cpu().numpy()


    # def evaluate(self, init, test_dataset, inverse_transform=True):
    #     """
    #     Prediction and forecast MSE evaluation.
    #     Prediction: reconstruct test set with test sensor measurements
    #     Forecast: reconstruct test set without test sensor measurements. Test sensor measurements
    #               are forecasted using Sindy or rolling forecaster.
    #     """
    #     if isinstance(self.latent_forecaster, SINDy_Forecaster):
    #         self.latent_forecaster.evaluate(init, test_dataset)
        
    #     # perform inverse transform


    #     # can call an evaluate method inside sindy_forecaster



def forecast(forecaster, reconstructor, test_dataset):
    initial_in = test_dataset.X[0:1].clone()
    vals = [initial_in[0, i, :].detach().cpu().clone().numpy() for i in range(test_dataset.X.shape[1])]
    for i in range(len(test_dataset.X)):
        scaled_output1, scaled_output2 = forecaster(initial_in)
        scaled_output1 = scaled_output1.detach().cpu().numpy()
        scaled_output2 = scaled_output2.detach().cpu().numpy()
        vals.append(np.concatenate([scaled_output1.reshape(test_dataset.X.shape[2]//2), scaled_output2.reshape(test_dataset.X.shape[2]//2)]))
        temp = initial_in.clone()
        initial_in[0, :-1] = temp[0, 1:]
        initial_in[0, -1] = torch.tensor(np.concatenate([scaled_output1, scaled_output2]))
    device = 'cuda' if next(reconstructor.parameters()).is_cuda else 'cpu'
    forecasted_vals = torch.tensor(np.array(vals), dtype=torch.float32).to(device)
    reconstructions = []
    for i in range(len(forecasted_vals) - test_dataset.X.shape[1]):
        recon = reconstructor(forecasted_vals[i:i + test_dataset.X.shape[1]].reshape(1, test_dataset.X.shape[1], test_dataset.X.shape[2])).detach().cpu().numpy()
        reconstructions.append(recon)
    reconstructions = np.array(reconstructions)
    return forecasted_vals, reconstructions
