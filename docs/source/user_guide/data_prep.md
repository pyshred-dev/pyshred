# Data Preparation

The `DataManager` object is in charge of handling data preparation and splitting.

## Import DataManager

`from pyshred import DataManager`


## Initialize Data Manager
The `DataManager` object takes in `lags` (int), `train_size` (float), `val_size` (float), and `test_size` (float). `Lags` represents the length of each sensor sequence being fed into the sequence model. Each sequence covers the timesteps `s(t-lags+1) ... s(t)`, ending at timestep `t`. The model uses this sequence to reconstruct the full state at `t`. `train_size`, `val_size`, and `test_size` are the proportions used to split the data into train, validation, and test datasets. These fields three arguments must sum up to 1.0.

Example:
Each input will be a sequence of 52 sensor measurement timesteps. 
```
manager = DataManager(
    lags=52, # build lagged sequences of 52 sensor measurement
    train_size=0.8, # use 80% of the data for training
    val_size=0.1, # use 10% of the data for validation
    test_size=0.1 # use 10% of the data for testing
)
```


## Add Data
The `add_data` method takes in:
- `data`: takes in a file path (.npy or .npz), a numpy array, or a torch.Tensor. The data should be array-like with **time on the first axis (axis 0)** followed by the spatial axes.
- `id`: a str to identify the dataset by. Each dataset must have a unique `id`.
- `random (optional)`: number of randomly placed sensors
- `stationary (optional)`: a list of stationary sensor locations. Each stationary sensor location should be a tuple representing its spatial coordinates.
- `mobile (optional)`: a list of mobile sensor locations. Each mobile sensor location should be a list representing its spatial coordinates at each timestep. The length of each mobile sensor location list should be equivalent to the size of the length of the first axis (axis 0) in `data`.
 - `measurements (optional)`: a 2D NumPy array (T, s) of raw sensor measurements
 - `compress (optional)`: the number of SVD components/POD modes to keep for `data`. Set to `False` or `None` for no compression.

Example:
```
manager.add_data(
    data="sst_data.npy", # path to sea-surface temperature data of shape (1400, 180, 360)
    id="SST", # dataset unique identifier set as "SST"
    random=3, # use three randomly placed sensors
    compress=50, # use 50 POD modes
)
```

To add additional datasets (e.g. different fields), call `add_data` for each additional dataset.



## Train/Validation/Test Split
The `prepare` method generates a train, validation, and test dataset.

Example:
```
train_dataset, val_dataset, test_dataset= manager.prepare()
```

## Useful/Common Attributes
- `sensor_summary_df`: a dataframe on sensor information with sensors as rows and sensor features/descriptors as columns.
- `sensor_measurements_df`: a dataframe of raw sensor measurments, with time as rows and sensors as columns.
- `train_sensor_measurements`: an array of sensor measurements (T,s) used for training
- `val_sensor_measurements`: an array of sensor measurements (T,s) used for validation
- `test_sensor_measurements`: an array of sensor measurements (T,s) used for testing


## Parametric Data
Use `ParametricDataManager` when the data holds many trajectories, e.g. simulations run with different parameter values or initial conditions. The data should have **trajectories on the first axis (axis 0), time on the second axis (axis 1)**, followed by the spatial axes. The data is split into train, validation, and test sets by trajectory, and each sensor sequence stays within a single trajectory.

### Known parameters
If the parameters of each trajectory are known, pass them to `add_data` with `params`. They are scaled and appended to the sensor measurements, so SHRED receives both as input.
- `params`: an array of shape (ntrajectories, ntimes, nparams), or (ntrajectories, nparams) for parameters that are constant in time.

Parameters are shared by every dataset in the manager, so provide them on only one `add_data` call.

Example:
```
manager = ParametricDataManager(lags=20, train_size=0.8, val_size=0.1, test_size=0.1)
manager.add_data(
    data=U, # shape (ntrajectories, ntimes, nx, ny)
    id="U",
    random=3,
    compress=4,
    params=MU, # shape (ntrajectories, ntimes, nparams)
)
manager.add_data(data=V, id="V", compress=4) # V shares the params provided with U
```

The raw parameters of each split are available as `train_params`, `val_params`, and `test_params`. Pass them to the `ParametricSHREDEngine` together with the sensor measurements:
```
engine.sensor_to_latent(manager.test_sensor_measurements, params=manager.test_params)
engine.evaluate(manager.test_sensor_measurements, test_Y, params=manager.test_params)
```

### Unknown parameters
To estimate parameters that are not known, add them as a dataset instead. SHRED then reconstructs them from the sensor measurements like any other field:
```
manager.add_data(data=MU, id="MU", compress=False)
```
