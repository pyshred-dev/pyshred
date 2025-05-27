from ..models.shred import SHRED
from ..processor.data_manager import DataManager
import numpy as np
import torch
import pandas as pd
from typing import Union, Dict
from ..processor.utils import *
from ..latent_forecaster_models.sindy import SINDy_Forecaster
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class SHREDEngine:
    # We want forecaster in the init as well... so need to fit in shred_model??
    def __init__(self, data_manager: DataManager, shred_model: SHRED):
        """
        data_manager   : DataManager (already .prepare()'d, so sensor_scaler exists)
        shred_model    : your trained SHRED instance
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
        # 3) Build lagged windows (shape -> (T, lags, n_sensors))
        lags = self.dm.lags
        lagged = generate_lagged_sensor_measurements(scaled_sensor_measurements, lags)
        # 4) To torch on same device as model:
        device = next(self.model.parameters()).device
        X = torch.tensor(lagged, dtype=torch.float32, device=device)
        # 5) Run through sequence to get latent:
        with torch.no_grad():
            # assumes your model.seq_model_outputs returns shape (T, latent_dim) when sindy=False
            latents = self.model.seq_model_outputs(X, sindy=False)
            # latents is a torch.Tensor shape (T, latent_dim)
        # 6) Return as numpy
        return latents.cpu().numpy()

    def forecast_latent(self, t, init_latents):

        ### move all model specific logic into the model itself
        ### pass in exactly t and init_latents to each model
        ### this wrapper should only help decide what model to go with
        if self.model.latent_forecaster is None:
            raise RuntimeError("No `latent_forecaster` available. Please initialize SHRED with a " \
            "`latent_forecaster` model.")
        if isinstance(init_latents, torch.Tensor):
            init_latents = init_latents.detach().cpu().numpy()
        # if isinstance(self.model.latent_forecaster, SINDy_Forecaster):
        return self.model.latent_forecaster.forecast(t, init_latents)
            # dt = self.model.latent_forecaster.dt
            # t_train = np.arange(0, t*dt, dt)
            # if init_latents.ndim > 2:
            #     raise ValueError(
            #         f"Invalid `init_latents`: expected a 1D array (shape (m,)) or a 2D array "
            #         f"(shape (timesteps, m)), but got a {init_latents.ndim}D array with shape {init_latents.shape}."
            #     )
            # if init_latents.ndim == 2:
            #     warnings.warn(
            #         f"`init_latents` has shape {init_latents.shape}; only its last row "
            #         "will be used as the initial latent state for SINDy.",
            #         UserWarning
            #     )
            #     init_latents = init_latents[-1]
            # return self.model.latent_forecaster.model.simulate(init_latents, t_train)
        

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


    def evaluate(
        self,
        sensor_measurements: np.ndarray,
        Y: Dict[str, np.ndarray]  # raw full‐state, exactly like decode() returns
    ) -> pd.DataFrame:
        """
        Performs end‐to‐end reconstruction error in the *physical* space.
        
        Parameters
        ----------
        sensor_measurements : (T, n_sensors)
            The test sensor time series.
        Y : dict[id] -> array (T, *spatial_shape)
            The *raw* full‐state ground truth for each dataset id.
        
        Returns
        -------
        DataFrame indexed by dataset id with columns [MSE, RMSE, MAE, R2].
        """
        # 1) Get the model’s reconstruction in raw space
        latents = self.sensor_to_latent(sensor_measurements)
        recon_dict = self.decode(latents)   # dict[id] -> (T, *spatial_shape)
        
        # 2) Compute stats
        records = []
        for id, y_true in Y.items():
            y_pred = recon_dict[id]
            y_true_flat = y_true.reshape(y_true.shape[0], -1)
            y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)

            mse   = mean_squared_error(y_true_flat, y_pred_flat)
            rmse  = np.sqrt(mse)
            mae   = mean_absolute_error(y_true_flat, y_pred_flat)
            r2    = r2_score(y_true_flat, y_pred_flat)

            records.append({
                "dataset": id, "MSE": mse, "RMSE": rmse,
                "MAE": mae, "R2": r2
            })
        return pd.DataFrame.from_records(records).set_index("dataset")