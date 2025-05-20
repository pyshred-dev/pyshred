import pysindy as ps
import torch
from ..models.sindy_utils import library_size
import torch.nn as nn

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


    def fit(self, latents):
        self.model.fit(latents, t=self.dt)