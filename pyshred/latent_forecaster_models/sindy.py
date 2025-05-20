import pysindy as ps
import itertools
import torch
import torch.nn.functional as F
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference
from ..models.sindy import sindy_library_torch
from ..models.sindy_utils import library_size
import torch.nn as nn
from typing import Optional

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
    
    # def initialize(self):
        # initialize coefficient matrix and mask





    # def __init__(self, latents, dt, poly_order=1, optimizer = ps.STLSQ(threshold=0.0, alpha=0.05), diff_method = ps.differentiation.FiniteDifference()):
    #     self.model = ps.SINDy(
    #         optimizer = optimizer,
    #         differentiation_method = diff_method,
    #         feature_library = ps.PolynomialLibrary(degree=poly_order)
    #     )
    #     self.model.fit(latents, t=dt)

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
        # self.model.fit(latents, t=self.dt)

    def fit(self, latents):
        self.model.fit(latents, t=self.dt)

def evaluate(self, init, test_dataset, inverse_transform=True):
    pass

# thoughts:
# model only performs prediction error
# forecasting error requires a forecasting model which we fit later