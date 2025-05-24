import pysindy as ps
import torch
from ..models.sindy_utils import library_size
import torch.nn as nn
import numpy as np
import warnings


class SINDy_Forecaster(nn.Module):
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

    def forecast(self, t, init_latents):
        dt = self.dt
        t_train = np.arange(0, t*dt, dt)
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