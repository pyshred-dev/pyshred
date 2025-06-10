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

class ParametricDataManager:
    """
    ParametricDataManager performs all processing needed to prepare the data for SHRED:
    - Gets sensor measurements from inputed sensor locations
    - Scales sensor measurements
    - Generates lagged sensor measurements
    - Scales full-state data
    - Compresses full-state data
    - Splits data into train, validation, and test

    Notes:
    ParametricDataManager is passed into the initialization of SHREDEngine for performing downstream tasks.
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
        self.sensor_scaler = None
        self.data_scaler = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
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
                 compress: Union[None, bool, int] = True, params: Optional[np.ndarray] = None, seed: Optional[int] = None):
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
        params: np.ndarray, optional
            Parameters for the dataset of shape (ntrajectories, ntimes, nparams)
        seed: int, optional
            Seed for selecting random sensor locations.
        """
        if id in self._dataset_ids:
            raise ValueError(f"Dataset id {id!r} already exists. Please choose a new id.")
        modes = parse_compress(compress)
        data = get_data(data)

        dataset_spatial_shape = data.shape[2:]
        ntimes = data.shape[1]
        ntrajectories = data.shape[0]

        train_indices = self.train_indices if self.train_indices is not None else np.arange(0, int(ntrajectories*self.train_size))
        val_indices = self.val_indices if self.val_indices is not None else np.arange(int(ntrajectories*self.train_size),
                                                    int(ntrajectories*self.train_size + ntrajectories*self.val_size))
        test_indices = self.test_indices if self.test_indices is not None else np.arange(int(ntrajectories*self.train_size +ntrajectories*self.val_size), ntrajectories)

        new_sensor_measurements_df_all_trajectories = None
        new_sensor_summary_df_all_trajectories = None
        new_sensor_measurements_all_trajectories = None
        for i in range(ntrajectories):
            traj_data = data[i]
            if measurements is not None:
                traj_meas = measurements[i]
            else:
                traj_meas = None
            if i == 0 and random is not None:
                if seed is None:
                    # initialize a seed so random sensor locations are the same across every trajectory
                    seed = np.random.randint(0, 1000000)
            sensor_start = next(self._sensor_number) if i == 0 else sensor_start
            # use temp sensor number to ensure sensor numbers are consistent across trajectories
            temp_sensor_number = count(start=sensor_start)
            sensors_dict = get_sensor_measurements(
                            data = traj_data,
                            id = id,
                            sensor_number = temp_sensor_number,
                            random = random,
                            stationary = stationary,
                            mobile = mobile,
                            measurements = traj_meas,
                            seed = seed
                        )
            # Advance the global counter only once
            if i == 0:
                # Count how many sensors were actually used
                num_sensors_used = len(sensors_dict['sensor_summary'])
                # Advance the global counter by the number of sensors used minus 1 
                # (since we already advanced it once above)
                for _ in range(num_sensors_used - 1):
                    next(self._sensor_number)
            new_sensor_measurements_df = sensors_dict['sensor_measurements_df']
            new_sensor_summary_df = sensors_dict['sensor_summary']
            new_sensor_measurements = sensors_dict['sensor_measurements']
            if new_sensor_measurements is not None:
                new_sensor_measurements = new_sensor_measurements[np.newaxis, :]

            if new_sensor_measurements_df_all_trajectories is None:
                new_sensor_measurements_df_all_trajectories = new_sensor_measurements_df
            else:
                new_sensor_measurements_df_all_trajectories = pd.concat([new_sensor_measurements_df_all_trajectories, new_sensor_measurements_df], axis=0, ignore_index=True)

            if new_sensor_summary_df_all_trajectories is None:
                new_sensor_summary_df_all_trajectories = new_sensor_summary_df

            if new_sensor_measurements_all_trajectories is None:
                new_sensor_measurements_all_trajectories = new_sensor_measurements
            else:
                new_sensor_measurements_all_trajectories = np.vstack((new_sensor_measurements_all_trajectories, new_sensor_measurements))

        train_data = data[train_indices]
        train_data = train_data.reshape(-1, np.prod(dataset_spatial_shape)) # flatten to a single spatial dimension
        val_data = data[val_indices]
        val_data = val_data.reshape(-1, np.prod(dataset_spatial_shape)) # flatten to a single spatial dimension
        test_data = data[test_indices]
        test_data = test_data.reshape(-1, np.prod(dataset_spatial_shape)) # flatten to a single spatial dimension

        if modes > 0:
            sc = StandardScaler()
            sc.fit(train_data)
            train_data = sc.transform(train_data)
            U, S, Vt = randomized_svd(train_data, n_components=modes, n_iter='auto')
            train_data = train_data @ Vt.T
            val_data = sc.transform(val_data)
            val_data = val_data @ Vt.T
            test_data = sc.transform(test_data)
            test_data = test_data @ Vt.T

        # commit to self
        self._dataset_ids.append(id)
        self._dataset_spatial_shape[self._dataset_ids[-1]] = dataset_spatial_shape
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        if self.sensor_measurements_df is None:
            self.sensor_measurements_df = new_sensor_measurements_df_all_trajectories
        else:
            self.sensor_measurements_df = pd.concat([self.sensor_measurements_df, new_sensor_measurements_df_all_trajectories], axis=1)
        if self.sensor_summary_df is None:
            self.sensor_summary_df = new_sensor_summary_df_all_trajectories
        else:
            self.sensor_summary_df = pd.concat([self.sensor_summary_df, new_sensor_summary_df_all_trajectories], axis = 0).reset_index(drop=True)
        if new_sensor_measurements_all_trajectories is not None:
            if self.sensor_measurements is None:
                self.sensor_measurements = new_sensor_measurements_all_trajectories
            else:
                self.sensor_measurements = np.concatenate(
                    (self.sensor_measurements, new_sensor_measurements_all_trajectories),
                    axis=-1
                )
            self.train_sensor_measurements = self.sensor_measurements[train_indices]
            self.val_sensor_measurements = self.sensor_measurements[val_indices]
            self.test_sensor_measurements = self.sensor_measurements[test_indices]
        if modes > 0:
            self._preSVD_scaler_registry[self._dataset_ids[-1]] = sc
            self._Vt_registry[self._dataset_ids[-1]] = Vt
        self._dataset_lengths[self._dataset_ids[-1]] = train_data.shape[1] # save spatial dim to registry
       
        if self.train_data is None:
            self.train_data = train_data
        else:
            self.train_data = np.hstack((self.train_data, train_data))
        
        if self.val_data is None:
            self.val_data = val_data
        else:
            self.val_data = np.hstack((self.val_data, val_data))
        
        if self.test_data is None:
            self.test_data = test_data
        else:
            self.test_data = np.hstack((self.test_data, test_data))

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
        sc.fit(self.train_data)
        self.data_scaler = sc
        scaled_train_data = sc.transform(self.train_data)
        scaled_val_data = sc.transform(self.val_data)
        scaled_test_data = sc.transform(self.test_data)


        if self.sensor_measurements is None:
            raise ValueError("No sensor measurements available. "
                            "Please call `add_data(...)` with sensor locations or sensor measurements.")

        train_sensor_measurements = self.sensor_measurements[self.train_indices]
        val_sensor_measurements = self.sensor_measurements[self.val_indices]
        test_sensor_measurements = self.sensor_measurements[self.test_indices]

        # flatten sensor measurements to shape (ntrajectories * ntimes, nsensors)
        flattened_train_sensor_measurements = train_sensor_measurements.reshape(-1, train_sensor_measurements.shape[-1])
        flattened_val_sensor_measurements = val_sensor_measurements.reshape(-1, val_sensor_measurements.shape[-1])
        flattened_test_sensor_measurements = test_sensor_measurements.reshape(-1, test_sensor_measurements.shape[-1])

        sc = MinMaxScaler()
        sc.fit(flattened_train_sensor_measurements)
        self.sensor_scaler = sc
        
        scaled_train_sensor_measurements = sc.transform(flattened_train_sensor_measurements)
        scaled_val_sensor_measurements = sc.transform(flattened_val_sensor_measurements)
        scaled_test_sensor_measurements = sc.transform(flattened_test_sensor_measurements)

        # reshape scaled sensor measurements back to shape (ntrajectories, ntimes, nsensors)
        scaled_train_sensor_measurements = scaled_train_sensor_measurements.reshape(train_sensor_measurements.shape)
        scaled_val_sensor_measurements = scaled_val_sensor_measurements.reshape(val_sensor_measurements.shape)
        scaled_test_sensor_measurements = scaled_test_sensor_measurements.reshape(test_sensor_measurements.shape)

        lagged_train_sensor_measurements = generate_lagged_sensor_measurements_rom(scaled_train_sensor_measurements, self.lags)
        lagged_val_sensor_measurements = generate_lagged_sensor_measurements_rom(scaled_val_sensor_measurements, self.lags)
        lagged_test_sensor_measurements = generate_lagged_sensor_measurements_rom(scaled_test_sensor_measurements, self.lags)

        device = get_device()

        # build torch tensors, dtype float32, on that device
        X_train = torch.tensor(lagged_train_sensor_measurements, dtype=torch.float32, device=device)
        Y_train = torch.tensor(scaled_train_data, dtype=torch.float32, device=device)

        X_val   = torch.tensor(lagged_val_sensor_measurements, dtype=torch.float32, device=device)
        Y_val   = torch.tensor(scaled_val_data, dtype=torch.float32, device=device)

        X_test  = torch.tensor(lagged_test_sensor_measurements, dtype=torch.float32, device=device)
        Y_test  = torch.tensor(scaled_test_data, dtype=torch.float32, device=device)

        train_dataset   = TimeSeriesDataset(X_train, Y_train)
        val_dataset     = TimeSeriesDataset(X_val, Y_val)
        test_dataset    = TimeSeriesDataset(X_test, Y_test)
        return train_dataset, val_dataset, test_dataset