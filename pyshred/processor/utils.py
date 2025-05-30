from typing import Union, List, Tuple, Optional
import numpy as np
import torch
import pandas as pd

DataInput = Union[
    str,                # .npy/.npz file path
    np.ndarray,         # raw NumPy
    torch.Tensor,       # PyTorch tensor
]


def get_data(data: DataInput) -> np.ndarray:
    """
    Takes in a file path (.npy or .npz), a numpy array, or a torch.Tensor.
    Returns a single numpy array.
    """
    if isinstance(data, str):
        if data.endswith('.npz'):
            return get_data_npz(data)
        elif data.endswith('.npy'):
            return get_data_npy(data)
        else:
            raise ValueError(f"Unsupported file format: {data}. Only .npy and .npz files are supported.")

    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()

    if isinstance(data, np.ndarray):
        return data

    raise ValueError(
        f"Unsupported input type: {type(data)}. "
        "Only .npy/.npz paths, numpy arrays, or torch.Tensor are supported."
    )


def get_data_npz(file_path: str) -> np.ndarray:
    data = np.load(file_path)
    if len(data.files) == 0:
        raise ValueError(f"The .npz file '{file_path}' is empty.")
    if len(data.files) > 1:
        raise ValueError(f"The .npz file '{file_path}' has multiple arrays: {data.files}.")
    return data[data.files[0]]


def get_data_npy(file_path: str) -> np.ndarray:
    return np.load(file_path)


def get_sensor_measurements(data, id, sensor_number, random, stationary, mobile, measurements):
    """
    - full_state_data: a nd numpy array with time on the first axis (axis 0)
    - random: number of randomly placed stationary sensors (integer)
    - stationary: coordinates of stationary sensors. Each sensor coordinate is a tuple.
                        If multiple stationary sensors, put tuples in a list (tuple or list of tuples).
    - mobile: coordinates (tuple) of a mobile sensor in a list (length of list should match number of timesteps in dataset).
                        if multiple mobile, use a nested list. (list of tuples, or nested list of tuples)
    - measurements: a NumPy array with time on axis 0 and sensors on axis 1
    Out:
    dict:
    {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
        "sensor_measurements_df": sensor_measurements_df,
    }
    where sensor_measurements is a 2D numpy array with time on axis 0
    and sensor_summary is a pandas dataframe
    """
    sensor_summary = []
    sensor_measurements = []

    # randomly selected stationary sensors
    if random is not None:
        if isinstance(random, int):
            random_sensor_locations = generate_random_sensor_locations(data = data, num_sensors = random)
            for sensor_coordinate in random_sensor_locations:
                sensor_summary.append([id, next(sensor_number), 'stationary (random)', sensor_coordinate])
                sensor_measurements.append(data[(slice(None),) + sensor_coordinate]) # all timesteps at that (i,j) location
        else:
            raise ValueError(
                f"`random` must be an int or None, "
                f"got {random!r} (type {type(random).__name__})"
            )

    # selected stationary sensors
    if stationary is not None:
        if isinstance(stationary, tuple):
            stationary = [stationary]
        if all(isinstance(sensor, tuple) for sensor in stationary):
            for sensor_coordinate in stationary:
                sensor_summary.append([id, next(sensor_number), 'stationary', sensor_coordinate])
                sensor_measurements.append(data[(slice(None),) + sensor_coordinate])
        else:
            raise ValueError(f"`stationary` must be a tuple or list of tuples or None, "
                             f"got {stationary!r} (type {type(stationary).__name__}).")

    # mobile sensors
    if mobile is not None:
        if isinstance(mobile[0], tuple):
            mobile = [mobile]
        if all(isinstance(ms, list) for ms in mobile):
            for mobile_sensor_coordinates in mobile:
                if len(mobile_sensor_coordinates) != data.shape[0]:
                    raise ValueError(
                        f"Number of mobile sensor coordinates ({len(mobile_sensor_coordinates)}) "
                        f"must match the number of timesteps ({data.shape[0]})."
                    )
                sensor_summary.append([id, next(sensor_number), 'mobile', mobile_sensor_coordinates])
                sensor_measurements.append([
                    data[timestep][sensor_coordinate]
                    for timestep, sensor_coordinate in enumerate(mobile_sensor_coordinates)
                ])
        else:
            raise ValueError(f"`mobile` must be a list of tuples or a list of list of tuples or None, "
                             f"got {mobile!r} (type {type(mobile).__name__}).")

    # sensor data
    if measurements is not None:
        if not isinstance(measurements, np.ndarray):
            raise TypeError(
                f"`measurements` must be a NumPy array or None, "
                f"got {measurements!r} (type {type(measurements).__name__})."
        )
        if measurements.shape[0] != data.shape[0]:
            raise ValueError(
                f"`measurements` is length {measurements.shape[0]}, expected {data.shape[0]}."
            )
        measurements = measurements.T
        for row in measurements:
            sensor_summary.append([id, next(sensor_number), 'measurement', None])
            sensor_measurements.append(list(row))

    # transpose sensor_measurements so time up on axis 0, number of sensors on axis 1
    if len(sensor_summary) == 0:
        sensor_summary = pd.DataFrame(columns=["data id", "sensor_number", "type", "loc/traj"])
    else:
        sensor_summary = pd.DataFrame(sensor_summary, columns=["data id", "sensor_number", "type", "loc/traj"])

    if len(sensor_measurements) == 0:
        sensor_measurements = None
        sensor_measurements_df = None
    else:
        sensor_measurements = np.array(sensor_measurements).T
        labels = (
            sensor_summary['data id']
            .astype(str)
            .str.cat(sensor_summary['sensor_number'].astype(str), sep='-')
        )
        labels.name = None
        sensor_measurements_df = pd.DataFrame(sensor_measurements, columns = labels)
    return {
        "sensor_measurements": sensor_measurements,
        "sensor_summary": sensor_summary,
        "sensor_measurements_df": sensor_measurements_df
    }

def generate_random_sensor_locations(data, num_sensors):
    """
    Generates a list of sensor locations given the number of sensors and
    a data numpy array where the first axis is time, followed by spatial axes.
    """
    spatial_shape = data.shape[1:] # first dimension is number of timesteps, rest is spatial dimentions
    spatial_points = np.prod(spatial_shape)
    sensor_indices = np.random.choice(spatial_points, size = num_sensors, replace = False)
    sensor_locations = []
    for sensor_index in sensor_indices:
        sensor_location = []
        for dim in reversed(spatial_shape):
            sensor_location.append(sensor_index % dim)
            sensor_index //= dim
        sensor_location = tuple(reversed(sensor_location))
        sensor_locations.append(sensor_location)
    return sensor_locations

def generate_lagged_sensor_measurements(sensor_measurements, lags):
    """
    Generates lagged sequences from sensor_measurments.
    Expects self.transformed_sensor_measurements to be a 2D numpy array with time is axis 0.
    Returns 3D numpy array of lagged sequences with timesteps along axis 0, lags along axis 1, sensors along axis 2.
    """
    num_timesteps = sensor_measurements.shape[0]
    num_sensors = sensor_measurements.shape[1]
    # concatenate zeros padding at beginning of sensor data along axis 0
    sensor_measurements = np.concatenate((np.zeros((lags, num_sensors)), sensor_measurements), axis = 0)
    lagged_sequences = np.empty((num_timesteps, lags, num_sensors))
    for i in range(lagged_sequences.shape[0]):
        lagged_sequences[i] = sensor_measurements[i:i+lags, :]
    return lagged_sequences
