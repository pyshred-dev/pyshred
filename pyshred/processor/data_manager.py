from typing import Union, List, Tuple, Optional
import numpy as np
import torch
from .utils import *
from ..objects.dataset import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
import warnings
from itertools import count

DEFAULT_MODES = 50

class DataManager:
    """
    DataManager performs all processing needed to prepare the data for SHRED:
    - Gets sensor measurements from inputed sensor locations
    - Scales sensor measurements
    - Generates lagged sensor measurements
    - Scales full-state data
    - Compresses full-state data
    - Splits data into train, validation, test, and holdout sets

    Notes:
    DataManager is passed into SHRED functions for training and evaluation.
    DataManager is passed into the initialization of SHREDEngine for performing downstream tasks.
    """
    def __init__(self, lags: int = 20, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1, holdout_size: float = 0.1):
        """
        lags : int
            The number of time steps to look back when building sensor sequences.
        train_size : float
            The fraction of the dataset (excluding holdout) to allocate for training.
        val_size : float
            The fraction of the dataset (excluding holdout) to allocate for validation.
        test_size : float
            The fraction of the dataset (excluding holdout) to allocate for testing reconstruction performance.
        holdout : float
            The fraction of the dataset to allocate for testing prediction and forecasting performance.
        """
        self.lags = lags
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.holdout_size = holdout_size
        self.cutoff_index = None
        self.sensor_summary_df = None
        self.sensor_measurements_df = None
        self.sensor_measurements = None
        self.data = None
        self.sensor_scaler = None
        self.data_scaler = None

        self._dataset_ids = []
        self._dataset_lengths = {}
        self._Vt_registry = {}
        self._preSVD_scaler_registry = {}
        self._sensor_number = count(start=0)

    def add_data(self, data: DataInput, id: str, random: Optional[int] = None,
                 stationary: Optional[Union[Tuple, List[Tuple]]] = None,
                 mobile: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
                 measurements: Optional[Union[List[float], List[List[float]]]] = None,
                 compress: Union[bool, int] = True, scale: bool = True):
        modes = self._parse_compress(compress)
        data = get_data(data)
        self._dataset_ids.append(id)
        sensors_dict = get_sensor_measurements(
                        data = data,
                        id = id,
                        sensor_number = self._sensor_number,
                        random = random,
                        stationary = stationary,
                        mobile = mobile,
                        measurements = measurements,
                    )
        new_sensor_measurements_df = sensors_dict['sensor_measurements_df']
        new_sensor_summary_df = sensors_dict['sensor_summary']
        new_sensor_measurements = sensors_dict['sensor_measurements']
        if self.sensor_measurements_df is None:
            self.sensor_measurements_df = new_sensor_measurements_df
        else:
            self.sensor_measurements_df = pd.concat([self.sensor_measurements_df, new_sensor_measurements_df], axis=1)
        if self.sensor_summary_df is None:
            self.sensor_summary_df = new_sensor_summary_df
        else:
            self.sensor_summary_df = pd.concat([self.sensor_summary_df, new_sensor_summary_df], axis = 0).reset_index(drop=True)
        if new_sensor_measurements is not None:
            if self.sensor_measurements is None:
                self.sensor_measurements = new_sensor_measurements
            else:
                self.sensor_measurements = np.hstack((self.sensor_measurements, new_sensor_measurements))

        if self.cutoff_index is None:
            cutoff_index = int(len(data)*(1-self.holdout_size))
            # Latent forecasting (e.g. SINDySHRED) does not support permutation of indices
            # holdin_indices = np.random.permutation(cutoff_index) 
            holdin_indices = np.arange(0,cutoff_index)
            self.train_indices = holdin_indices[:int(len(holdin_indices)*self.train_size)]
            self.val_indices = holdin_indices[int(len(holdin_indices)*self.train_size):int(len(holdin_indices)*self.train_size + len(holdin_indices)*self.val_size)]
            self.test_indices = holdin_indices[int(len(holdin_indices)*self.train_size + len(holdin_indices)*self.val_size):]
            self.holdout_indices = np.arange(cutoff_index,len(data))

        if modes > 0:
            sc = StandardScaler()
            sc.fit(data[self.train_indices])
            self._preSVD_scaler_registry[self._dataset_ids[-1]] = sc # save scaler to registery
            data = sc.transform(data)
            U, S, Vt = randomized_svd(data[self.train_indices], n_components=modes, n_iter='auto')
            data = data @ Vt.T
            self._Vt_registry[self._dataset_ids[-1]] = Vt # save V transpose to registry
        self._dataset_lengths[self._dataset_ids[-1]] = data.shape[1] # save spatial dim to registry

        if self.data is None:
            self.data = data
        else:
            self.data = np.hstack((self.data, data))


    def prepare(self):
        sc = MinMaxScaler()
        sc.fit(self.data[self.train_indices])
        self.data_scaler = sc
        scaled_data = sc.transform(self.data)
        if self.sensor_measurements is None:
            raise ValueError("No sensor measurements available. "
                            "Please call `add_data(...)` with sensor locations or sensor measurements.")
        sc = MinMaxScaler()
        sc.fit(self.sensor_measurements[self.train_indices])
        self.sensor_scaler = sc
        scaled_sensor_measurements = sc.transform(self.sensor_measurements)
        lagged_sensor_measurements = generate_lagged_sensor_measurements(scaled_sensor_measurements, self.lags)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # build torch tensors, dtype float32, on that device
        X_train = torch.tensor(lagged_sensor_measurements[self.train_indices], dtype=torch.float32, device=device)
        Y_train = torch.tensor(scaled_data[self.train_indices], dtype=torch.float32, device=device)

        X_val   = torch.tensor(lagged_sensor_measurements[self.val_indices], dtype=torch.float32, device=device)
        Y_val   = torch.tensor(scaled_data[self.val_indices], dtype=torch.float32, device=device)

        X_test  = torch.tensor(lagged_sensor_measurements[self.test_indices], dtype=torch.float32, device=device)
        Y_test  = torch.tensor(scaled_data[self.test_indices], dtype=torch.float32, device=device)

        X_hold  = torch.tensor(lagged_sensor_measurements[self.holdout_indices], dtype=torch.float32, device=device)
        Y_hold  = torch.tensor(scaled_data[self.holdout_indices], dtype=torch.float32, device=device)

        train_dataset   = TimeSeriesDataset(X_train, Y_train)
        val_dataset     = TimeSeriesDataset(X_val, Y_val)
        test_dataset    = TimeSeriesDataset(X_test, Y_test)
        holdout_dataset = TimeSeriesDataset(X_hold, Y_hold)
        return train_dataset, val_dataset, test_dataset, holdout_dataset


    def _parse_compress(self, compress: Union[bool, int]) -> int:
        """Normalize the compress argument into an integer number of modes."""
        if isinstance(compress, bool):
            return DEFAULT_MODES if compress else 0
        if isinstance(compress, int):
            return compress
        raise TypeError(f"`compress` must be bool or int, got {type(compress).__name__!r}")