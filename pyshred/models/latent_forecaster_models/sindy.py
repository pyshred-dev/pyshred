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
    Sparse Identification of Nonlinear Dynamics for latent space.
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


    def initialize(self, latent_dim):
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
        self.model.fit(latents, t=self.dt)

    def __str__(self, lhs=None, precision=3):
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