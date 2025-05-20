# # normalize latent outputs between [-1,1]
# from sklearn.preprocessing import MinMaxScaler

# # assume `Z` is your (N × k) latent array, either np.ndarray or torch.Tensor converted to np
# scaler = MinMaxScaler(feature_range=(-1, 1))
# Z_norm = scaler.fit_transform(Z)

# Z_val_norm = scaler.transform(Z_val)
# Z_unscaled = scaler.inverse_transform(Z_norm)

# # Torch
# mins, _ = Z.min(dim=0)
# maxs, _ = Z.max(dim=0)
# Z0_1 = (Z - mins) / (maxs - mins)    # now in [0,1]
# Zm1_1 = 2 * Z0_1 - 1                 # now in [-1,1]

#  fit Sindy

#  simulate sindy with an initial seed (latent space at a single timestep)


# SHREDEngine(DataManager, SHRED)
# encode_sensors(sensor_measurements) -> latent_space
# encode_forecast(t, initialization = None) -> latent_space # can get initialization from self.encoder_sensors(...)
# decode(latent_space)

from ..models.sindy_shred import SINDy_SHRED
from ..processor.data_manager import DataManager
import numpy as np
import torch
import pandas as pd
from typing import Union
from ..processor.utils import *

class SHREDEngine:
    # We want forecaster in the init as well... so need to fit in shred_model??
    def __init__(self, data_manager: DataManager, shred_model: SINDy_SHRED):
        """
        data_manager   : DataManager (already .prepare()'d, so sensor_scaler exists)
        shred_model    : your trained SINDy_SHRED instance
        """
        self.dm    = data_manager
        self.model = shred_model
        # ensure model is in eval mode
        self.model.eval()


    def sensor_to_latent(self, sensor_measurements: Union[np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
        """
        Convert raw sensor measurements into latent-space embeddings.
        
        Parameters
        ----------
        sensor_measurements : array-like of shape (T, n_sensors)
            Raw sensor time series.

        Returns
        -------
        latents : np.ndarray of shape (T, latent_dim)
            The GRU/LSTM final-hidden-state at each time index.
        """
        # 1) Pull out raw numpy array
        if isinstance(sensor_measurements, pd.DataFrame):
            sensor_measurements = sensor_measurements.values
        elif torch.is_tensor(sensor_measurements):
            sensor_measurements = sensor_measurements.detach().cpu().numpy()
        elif isinstance(sensor_measurements, np.ndarray):
            sensor_measurements = sensor_measurements
        else:
            raise TypeError(f"Unsupported type {type(sensor_measurements)} for sensor_measurements")
        # 2) Scale using the DataManager's fitted scaler (shape -> (T, n_sensors))
        scaled_sensor_measurements = self.dm.sensor_scaler.transform(sensor_measurements)  
        # 3) Build lagged windows (shape -> (T, lags+1, n_sensors))
        lags = self.dm.lags
        lagged = generate_lagged_sensor_measurements(scaled_sensor_measurements, lags)
        # 4) To torch on same device as model:
        device = next(self.model.parameters()).device
        X = torch.tensor(lagged, dtype=torch.float32, device=device)
        # 5) Run through sequence to get latent:
        with torch.no_grad():
            # assumes your model.gru_outputs returns shape (T, latent_dim) when sindy=False
            latents = self.model.gru_outputs(X, sindy=False)
            # latents is a torch.Tensor shape (T, latent_dim)
        # 6) Return as numpy
        return latents.cpu().numpy()


# recon_dict_out = manager.postprocess(reconstruction, mode = "reconstruct")
    def decode(self, latents):
        device = next(self.model.decoder.parameters()).device
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).to(device).float()
        else:
            latents = latents.to(device).float()
        self.model.decoder.eval()
        with torch.no_grad():
            output = self.model.decoder(latents)
        output = output.detach().cpu().numpy()
        output = self.dm.data_scaler.inverse_transform(output)
        results = {}
        start_index = 0
        for id in self.dm._dataset_ids:
            length = self.dm._dataset_lengths.get(id)
            Vt = self.dm._Vt_registry.get(id)
            preSVD_scaler = self.dm._preSVD_scaler_registry.get(id)
            spatial_shape = self.dm._dataset_spatial_shape.get(id)
            dataset = output[:,start_index:start_index+length]
            if Vt is not None:
                dataset = dataset @ Vt
            if preSVD_scaler is not None:
                dataset = preSVD_scaler.inverse_transform(dataset)
            original_shape = (dataset.shape[0],) + spatial_shape
            results[id] = dataset.reshape(original_shape)
            start_index = length + start_index
        return results


    def decode(self, latents):
        device = next(self.model.decoder.parameters()).device
        if isinstance(latents, np.ndarray):
            latents = torch.from_numpy(latents).to(device).float()
        else:
            latents = latents.to(device).float()
        decoder_input = {"final_hidden_state": latents} # DAVID this only works for SDN type latents not UNET
        self.model.decoder.eval()
        with torch.no_grad():
            output = self.model.decoder(decoder_input)
        output = output.detach().cpu().numpy()
        output = self.dm.data_scaler.inverse_transform(output)
        results = {}
        start_index = 0
        for id in self.dm._dataset_ids:
            length = self.dm._dataset_lengths.get(id)
            Vt = self.dm._Vt_registry.get(id)
            preSVD_scaler = self.dm._preSVD_scaler_registry.get(id)
            spatial_shape = self.dm._dataset_spatial_shape.get(id)
            dataset = output[:,start_index:start_index+length]
            if Vt is not None:
                dataset = dataset @ Vt
            if preSVD_scaler is not None:
                dataset = preSVD_scaler.inverse_transform(dataset)
            original_shape = (dataset.shape[0],) + spatial_shape
            results[id] = dataset.reshape(original_shape)
            start_index = length + start_index
        return results


    def evaluate_forecast(self, init, test_dataset, inverse_transform=True):
        # returns prediction and forecast MSE on test dataset
        # if inverse transform is True: undos any minmax scaling and compression
        # else: comparable error to train and validation MSE
        pass


    def evaluate_forecast(self, init_latents, Y, inverse_transform=True):
        pass


    def evaluate_reconstruction(self, sensor_measurements: Union[np.ndarray, torch.Tensor, pd.DataFrame], Y, inverse_transform=True):
        latent = self.sensor_to_latent(sensor_measurements)
