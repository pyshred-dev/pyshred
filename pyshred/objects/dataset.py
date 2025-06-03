import torch

class TimeSeriesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for time series sensor data and corresponding full-state measurements.

    Parameters
    ----------
    X : torch.Tensor
        Input sensor sequences of shape (batch_size, lags, num_sensors).
    Y : torch.Tensor
        Target full-state measurements of shape (batch_size, state_dim).

    Attributes
    ----------
    X : torch.Tensor
        Sensor measurement sequences.
    Y : torch.Tensor
        Full-state target measurements.
    len : int
        Number of samples in the dataset.
    """
    
    def __init__(self, X, Y):
        """
        Initialize the TimeSeriesDataset.

        Parameters
        ----------
        X : torch.Tensor
            Input sensor sequences of shape (batch_size, lags, num_sensors).
        Y : torch.Tensor
            Target full-state measurements of shape (batch_size, state_dim).
        """
        self.X = X
        self.Y = Y
        self.len = X.shape[0]

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (sensor_sequence, target_state) pair.
        """
        return self.X[index], self.Y[index]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.len


# class TimeSeriesDataset(torch.utils.data.Dataset):
#     """
#     Takes input sequence of sensor measurements with shape (batch_size, lags, num_sensors)
#     and corresponding measurements of high-dimensional state, returns a Torch Dataset.
#     """
#     def __init__(self, X=None, Y=None):
#         # X, Y should be torch.Tensor of shape (N, lags, num_sensors) and (N, ...state dims)
#         self.X = X
#         self.Y = Y
#         self.len = 0 if X is None else X.shape[0]

#     def add_data(self, X_new: torch.Tensor, Y_new: torch.Tensor):
#         """Append new (X_new, Y_new), or initialize if empty."""
#         if self.X is None:
#             # first time
#             self.X = X_new
#             self.Y = Y_new
#         else:
#             # append along the batch dimension
#             self.X = torch.cat([self.X, X_new], dim=0) # might be dim = 2
#             self.Y = torch.cat([self.Y, Y_new], dim=0) # might be dim = 1

#         self.len = self.X.shape[0]

#     def __getitem__(self, index):
#         return self.X[index], self.Y[index]

#     def __len__(self):
#         return self.len
