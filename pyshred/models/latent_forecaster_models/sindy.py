import pysindy as ps
import torch
from .._sindy_utils import library_size
import torch.nn as nn
import numpy as np
import warnings
try: 
    from pysindy.optimizers import SINDyPI
    sindy_pi_flag = True
except ImportError:
    sindy_pi_flag = False
from .abstract_latent_forecaster import AbstractLatentForecaster


class SINDy_Forecaster(AbstractLatentForecaster):
    """
    Sparse Identification of Nonlinear Dynamics (SINDy) forecaster for latent space dynamics.
    
    This forecaster uses the SINDy algorithm to discover governing equations for the latent
    space dynamics and uses these equations for forecasting. SINDy identifies sparse
    dynamical systems by finding a parsimonious balance between model complexity and
    prediction accuracy.

    Parameters
    ----------
    poly_order : int, optional
        Maximum polynomial order for the feature library. Defaults to 1.
    include_sine : bool, optional
        Whether to include sine functions in the feature library. Defaults to False.
    dt : float, optional
        Time step size for the dynamical system. Defaults to 1.0.
    optimizer : pysindy.optimizers, optional
        SINDy optimizer for sparse regression. Defaults to STLSQ with threshold=0.0 and alpha=0.05.
    diff_method : pysindy.differentiation, optional
        Method for computing time derivatives. Defaults to FiniteDifference().

    Attributes
    ----------
    seed_length : int
        Number of latent space timesteps required to seed forecaster (always 1 for SINDy).
    latent_dim : int
        Dimension of the latent space.
    lib_dim : int
        Dimension of the feature library after initialization.
    model : pysindy.SINDy
        The fitted SINDy model containing discovered equations.
    coefficients : torch.nn.Parameter
        Learnable coefficients tensor for the dynamical system.
    coefficient_mask : torch.Tensor
        Binary mask indicating active coefficients.
    """
    def __init__(self,
                 poly_order: int = 1,
                 include_sine: bool = False,
                 dt: float = 1.0,
                 optimizer = ps.STLSQ(threshold=0.0, alpha=0.05),
                 diff_method = ps.differentiation.FiniteDifference()):
        super().__init__()
        self.seed_length = 1 # number of latent space timesteps required to seed forecaster
        self.latent_dim = None
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.dt = dt
        self.optimizer = optimizer
        self.diff_method = diff_method
        self.model = None 


    def initialize(self, latent_dim):
        """
        Initialize the SINDy model with the given latent dimension.
        
        This method sets up the feature library dimension, initializes learnable
        parameters, and creates the SINDy model with the specified configuration.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        """
        self.latent_dim = latent_dim
        self.lib_dim = library_size(self.latent_dim, self.poly_order, self.include_sine)
        self.coefficients = nn.Parameter(torch.zeros(self.lib_dim, self.latent_dim))
        self.register_buffer('coefficient_mask', torch.zeros(self.lib_dim, self.latent_dim))
        self.model = ps.SINDy(
            optimizer = self.optimizer,
            differentiation_method = self.diff_method,
            feature_library = ps.PolynomialLibrary(degree=self.poly_order)
        )

    def forecast(self, h, init_latents):
        """
        Forecast the next h latent steps using the discovered SINDy equations.
        
        The forecasting is performed by simulating the discovered dynamical system
        forward in time from the given initial conditions.

        Parameters
        ----------
        h : int
            Number of time steps to forecast into the future.
        init_latents : array-like
            Initial latent state(s) for forecasting. Can be:
            - 1D array of shape (latent_dim,): single initial state
            - 2D array of shape (timesteps, latent_dim): uses last row as initial state

        Returns
        -------
        numpy.ndarray
            Forecasted latent states of shape (h, latent_dim).
            
        Raises
        ------
        ValueError
            If init_latents has more than 2 dimensions.
            
        Warns
        -----
        UserWarning
            If init_latents is 2D, only the last row will be used.
        """
        dt = self.dt
        t_train = np.arange(0, h*dt, dt)
        if init_latents.ndim > 2:
            raise ValueError(
                f"Invalid `init_latents`: expected a 1D array (shape (m,)) or a 2D array "
                f"(shape (timesteps, m)), but got a {init_latents.ndim}D array with shape {init_latents.shape}."
            )
        if init_latents.ndim == 2:
            warnings.warn(
                f"`init_latents` has shape {init_latents.shape}; only its last row "
                "will be used as the initial latent state for SINDy.",
                UserWarning
            )
            init_latents = init_latents[-1]
        return self.model.simulate(init_latents, t_train)

    def fit(self, latents):
        """
        Fit the SINDy model to the latent time series data.
        
        This method discovers the sparse dynamical system governing the latent
        space evolution by fitting the SINDy model to the provided data.

        Parameters
        ----------
        latents : array-like
            Latent time series data of shape (T, latent_dim) where T is the
            number of time steps and latent_dim is the dimension of the latent space.
        """
        self.model.fit(latents, t=self.dt)

    def __str__(self, lhs=None, precision=3):
        """
        Return a string representation of the SINDy forecaster.
        
        If the model is trained, returns the discovered equations. If untrained,
        returns the model configuration.

        Parameters
        ----------
        lhs : list, optional
            Custom left-hand side variable names for equations. If None, uses default names.
        precision : int, optional
            Number of decimal places for equation coefficients. Defaults to 3.

        Returns
        -------
        str
            String representation showing either configuration or discovered equations.
        """
        if not hasattr(self.model, 'n_features_in_') or self.model.n_features_in_ is None:
            info = []
            info.append(f"SINDy_Forecaster(")
            info.append(f"  poly_order={self.poly_order}")
            info.append(f"  include_sine={self.include_sine}")
            info.append(f"  dt={self.dt}")
            info.append(f"  optimizer={type(self.optimizer).__name__}")
            info.append(f"  diff_method={type(self.diff_method).__name__}")
            if self.latent_dim is not None:
                info.append(f"  latent_dim={self.latent_dim}")
                info.append(f"  lib_dim={getattr(self, 'lib_dim', 'Not initialized')}")
            else:
                info.append(f"  latent_dim=Not initialized")
            info.append(")")
            return "\n".join(info)
        eqns = self.model.equations(precision=precision)
        if sindy_pi_flag and isinstance(self.model.optimizer, SINDyPI):
            feature_names = self.model.get_feature_names()
        else:
            feature_names = self.model.feature_names
        lines = []
        for i, eqn in enumerate(eqns):
            if self.model.discrete_time:
                name = f"({feature_names[i]})"
                lines.append(f"{name}[k+1] = {eqn}")
            elif lhs is None:
                if not sindy_pi_flag or not isinstance(self.model.optimizer, SINDyPI):
                    name = f"({feature_names[i]})"
                    lines.append(f"{name}' = {eqn}")
                else:
                    lines.append(f"{feature_names[i]} = {eqn}")
            else:
                lines.append(f"{lhs[i]} = {eqn}")

        return "\n".join(lines)
    
    def __repr__(self):
        """
        Return a detailed representation of the SINDy forecaster.
        
        Shows the complete configuration of the forecaster including all parameters,
        initialization status, and model state.

        Returns
        -------
        str
            Detailed multi-line string representation of the forecaster configuration.
        """
        info = []
        info.append(f"SINDy_Forecaster(")
        info.append(f"  poly_order={self.poly_order}")
        info.append(f"  include_sine={self.include_sine}")
        info.append(f"  dt={self.dt}")
        info.append(f"  optimizer={type(self.optimizer).__name__}")
        info.append(f"  diff_method={type(self.diff_method).__name__}")
        if hasattr(self, 'latent_dim') and self.latent_dim is not None:
            info.append(f"  latent_dim={self.latent_dim}")
            info.append(f"  lib_dim={getattr(self, 'lib_dim', 'Not initialized')}")
        else:
            info.append(f"  latent_dim=Not initialized")
        info.append(")")
        return "\n".join(info)