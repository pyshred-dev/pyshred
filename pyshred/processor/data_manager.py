from typing import Union, List, Tuple, Optional
import numpy as np
import torch
from .utils import *
from ..objects.dataset import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.extmath import randomized_svd
from itertools import count
from ..objects.device import get_device

DEFAULT_MODES = 50

class DataManager:
    """
    DataManager performs all processing needed to prepare the data for SHRED:
    - Gets sensor measurements from inputed sensor locations
    - Scales sensor measurements
    - Generates lagged sensor measurements
    - Scales full-state data
    - Compresses full-state data
    - Splits data into train, validation, and test

    Notes:
    DataManager is passed into the initialization of SHREDEngine for performing downstream tasks.
    """
    def __init__(self, lags: int = 20, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1):
        """
        lags : int
            The number of past time steps (lags) included in each sensor input sequence
        train_size : float
            The fraction of the dataset to allocate for training.
        val_size : float
            The fraction of the dataset to allocate for validation.
        test_size : float
            The fraction of the dataset to allocate for testing reconstruction performance.
        """
        if not abs(train_size + val_size + test_size - 1.0) < 1e-8:
            raise ValueError("train_size, val_size, and test_size must sum to 1.0")
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.lags = lags

        self.sensor_summary_df = None
        self.sensor_measurements_df = None
        self.sensor_measurements = None
        self.data = None
        self.sensor_scaler = None
        self.data_scaler = None

        self._dataset_ids = []
        self._dataset_spatial_shape = {}
        self._dataset_lengths = {} # spatial dim after flattening to a single spatial axis
        self._Vt_registry = {}
        self._preSVD_scaler_registry = {}
        self._sensor_number = count(start=0)

        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
        self.train_sensor_measurements = None
        self.val_sensor_measurements = None
        self.test_sensor_measurements = None

    def add_data(self, data: DataInput, id: str, random: Optional[int] = None,
                 stationary: Optional[Union[Tuple, List[Tuple]]] = None,
                 mobile: Optional[Union[List[Tuple], List[List[Tuple]]]] = None,
                 measurements: Optional[np.ndarray] = None,
                 compress: Union[None, bool, int] = True,
                 seed: Optional[int] = None):
        """
        Add a dataset with sensor measurements to the data manager.

        Parameters
        ----------
        data : DataInput
            Input data as file path (.npy/.npz), numpy array, or torch tensor.
        id : str
            Unique identifier for this dataset.
        random : int, optional
            Number of randomly placed stationary sensors.
        stationary : tuple or list of tuples, optional
            Coordinates of stationary sensors.
        mobile : list of tuples or list of list of tuples, optional
            Coordinates of mobile sensors for each timestep.
        measurements : np.ndarray, optional
            Pre-computed sensor measurements with time on axis 0.
        compress : None, bool, or int, optional
            Data compression settings. True uses default modes, int specifies number of modes.
        seed : int, optional
            Seed for selecting random sensor locations.
        """
        if id in self._dataset_ids:
            raise ValueError(f"Dataset id {id!r} already exists. Please choose a new id.")
        modes = parse_compress(compress)
        data = get_data(data)
        dataset_spatial_shape = data.shape[1:]
        train_indices = self.train_indices if self.train_indices is not None else np.arange(0, int(len(data)*self.train_size))
        val_indices = self.val_indices if self.val_indices is not None else np.arange(int(len(data)*self.train_size),
                                                    int(len(data)*self.train_size + len(data)*self.val_size))
        test_indices = self.test_indices if self.test_indices is not None else np.arange(int(len(data)*self.train_size +len(data)*self.val_size), len(data))
        sensors_dict = get_sensor_measurements(
                        data = data,
                        id = id,
                        sensor_number = self._sensor_number,
                        random = random,
                        stationary = stationary,
                        mobile = mobile,
                        measurements = measurements,
                        seed = seed
                    )
        new_sensor_measurements_df = sensors_dict['sensor_measurements_df']
        new_sensor_summary_df = sensors_dict['sensor_summary']
        new_sensor_measurements = sensors_dict['sensor_measurements']
        data = data.reshape(data.shape[0], -1) # flatten to a single spatial dimension
        if modes > 0:
            sc = StandardScaler()
            sc.fit(data[train_indices])
            data = sc.transform(data)
            U, S, Vt = randomized_svd(data[train_indices], n_components=modes, n_iter='auto')
            data = data @ Vt.T
        # commit to self
        self._dataset_ids.append(id)
        self._dataset_spatial_shape[self._dataset_ids[-1]] = dataset_spatial_shape
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
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
            self.train_sensor_measurements = self.sensor_measurements[train_indices]
            self.val_sensor_measurements = self.sensor_measurements[val_indices]
            self.test_sensor_measurements = self.sensor_measurements[test_indices]
        if modes > 0:
            self._preSVD_scaler_registry[self._dataset_ids[-1]] = sc
            self._Vt_registry[self._dataset_ids[-1]] = Vt
        self._dataset_lengths[self._dataset_ids[-1]] = data.shape[1] # save spatial dim to registry
        if self.data is None:
            self.data = data
        else:
            self.data = np.hstack((self.data, data))

    def prepare(self):
        """
        Prepare the data for training by scaling and creating lagged sequences.

        Returns
        -------
        tuple
            (train_dataset, val_dataset, test_dataset) - PyTorch datasets ready for training.

        Raises
        ------
        ValueError
            If no sensor measurements are available.
        """
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

        device = get_device()

        # build torch tensors, dtype float32, on that device
        X_train = torch.tensor(lagged_sensor_measurements[self.train_indices], dtype=torch.float32, device=device)
        Y_train = torch.tensor(scaled_data[self.train_indices], dtype=torch.float32, device=device)

        X_val   = torch.tensor(lagged_sensor_measurements[self.val_indices], dtype=torch.float32, device=device)
        Y_val   = torch.tensor(scaled_data[self.val_indices], dtype=torch.float32, device=device)

        X_test  = torch.tensor(lagged_sensor_measurements[self.test_indices], dtype=torch.float32, device=device)
        Y_test  = torch.tensor(scaled_data[self.test_indices], dtype=torch.float32, device=device)

        train_dataset   = TimeSeriesDataset(X_train, Y_train)
        val_dataset     = TimeSeriesDataset(X_val, Y_val)
        test_dataset    = TimeSeriesDataset(X_test, Y_test)
        return train_dataset, val_dataset, test_dataset