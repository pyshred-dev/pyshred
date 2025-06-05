from ..models.shred import SHRED
from ..processor.parametric_data_manager import ParametricDataManager
import numpy as np
import torch
import pandas as pd
from typing import Union, Dict
from ..processor.utils import *
from ..models.latent_forecaster_models.sindy import SINDy_Forecaster
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ParametricSHREDEngine:
    """
    High-level interface for SHRED model inference and evaluation.

    Parameters
    ----------
    data_manager : ParametricDataManager
        Prepared data manager with fitted scalers.
    shred_model : SHRED
        Trained SHRED model instance.

    Attributes
    ----------
    dm : ParametricDataManager
        The data manager for preprocessing and postprocessing.
    model : SHRED
        The trained SHRED model.
    """
    
    def __init__(self, data_manager: ParametricDataManager, shred_model: SHRED):
        """
        Initialize the SHRED inference engine.

        Parameters
        ----------
        data_manager : ParametricDataManager
            Already prepared DataManager with fitted scalers.
        shred_model : SHRED
            Trained SHRED model instance.
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
        # 2) Handle different input shapes
        if len(sensor_measurements.shape) == 2:
            # Single trajectory (T, n_sensors) - add trajectory dimension
            sensor_measurements = sensor_measurements[np.newaxis, :]
        # Expected shape: (n_trajectories, T, n_sensors)
        if len(sensor_measurements.shape) != 3:
            raise ValueError(f"Expected input shape (T, n_sensors) or (n_trajectories, T, n_sensors), "
                             f"got {sensor_measurements.shape}")
        # 3) Scale using the DataManager's fitted scaler
        # Flatten to (n_trajectories * T, n_sensors) for scaling, then reshape back
        flattened_measurements = sensor_measurements.reshape(-1, sensor_measurements.shape[-1])
        scaled_flattened = self.dm.sensor_scaler.transform(flattened_measurements)
        scaled_measurements = scaled_flattened.reshape(sensor_measurements.shape)

        # 4) Generate lagged sequences for each trajectory
        lagged = generate_lagged_sensor_measurements_rom(scaled_measurements, self.dm.lags)
        
        # 5) To torch on same device as model:
        device = next(self.model.parameters()).device
        X = torch.tensor(lagged, dtype=torch.float32, device=device)
        
        # 6) Run through sequence to get latent:
        with torch.no_grad():
            # assumes your model._seq_model_outputs returns shape (T, latent_dim) when sindy=False
            latents = self.model._seq_model_outputs(X, sindy=False)
            # latents is a torch.Tensor shape (T, latent_dim)

        # 7) Return as numpy
        return latents.cpu().numpy()


    def decode(self, latents):
        """
        Decode latent states back to full physical state space.

        Parameters
        ----------
        latents : np.ndarray or torch.Tensor
            Latent representations to decode.

        Returns
        -------
        dict
            Dictionary mapping dataset IDs to reconstructed physical states.
        """
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
        # 1) Get the model's reconstruction in raw space
        latents = self.sensor_to_latent(sensor_measurements)
        recon_dict = self.decode(latents)   # dict[id] -> (ntrajectories*T, *spatial_shape)
        
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