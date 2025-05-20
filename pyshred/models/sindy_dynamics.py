import itertools
import torch
import torch.nn.functional as F
from pysindy import SINDy
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference
from .sindy import sindy_library_torch
from .sindy_utils import library_size
import torch.nn as nn
from typing import Optional

def _build_sindy_feature_names(n: int,
                              poly_order: int,
                              include_sine: bool = False,
                              include_constant: bool = True) -> list[str]:
    """
    Generate feature names matching sindy_library_torch ordering.
    """
    names = []
    if include_constant:
        names.append("1")
    # monomials
    for deg in range(1, poly_order + 1):
        for comb in itertools.combinations_with_replacement(range(n), deg):
            counts = {i: comb.count(i) for i in set(comb)}
            term = "*".join(
                f"z{i}^{counts[i]}" if counts[i] > 1 else f"z{i}"
                for i in sorted(counts)
            )
            names.append(term)
    # sine
    if include_sine:
        for i in range(n):
            names.append(f"sin(z{i})")
    return names


class SINDyDynamics(nn.Module):
    """
    Sparse Identification of Nonlinear Dynamics for latent space.
    """
    def __init__(self,
                 latent_dim: int,
                 poly_order: int = 3,
                 include_sine: bool = False,
                 dt: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.dt = dt
        # initialize coefficient matrix and mask
        self.lib_dim = library_size(latent_dim, poly_order, include_sine)
        self.coefficients = nn.Parameter(torch.zeros(self.lib_dim, latent_dim))
        self.register_buffer('coefficient_mask', torch.zeros(self.lib_dim, latent_dim))

    def fit(self,
            z: torch.Tensor,
            dt: Optional[float] = None,
            threshold: float = 1e-6):
        """
        Fit SINDy coefficients from latent trajectory z (T x latent_dim).
        Uses least-squares and hard thresholding.
        """
        dt = dt if dt is not None else self.dt
        # prepare data
        z_curr = z[:-1]
        z_next = z[1:]
        dz = (z_next - z_curr) / dt
        # build library
        Theta = sindy_library_torch(z_curr, self.latent_dim, self.poly_order, self.include_sine)
        # least-squares: Xi = pinv(Theta) @ dz
        Xi = torch.pinverse(Theta) @ dz
        # threshold small coefficients
        mask = (Xi.abs() > threshold).float()
        # store
        self.coefficients.data.copy_(Xi)
        self.coefficient_mask.copy_(mask)
        self.coefficients.data.mul_(mask)

    def predict_next(self,
                     z: torch.Tensor,
                     dt: Optional[float] = None) -> torch.Tensor:
        dt = dt if dt is not None else self.dt
        Theta = sindy_library_torch(z, self.latent_dim, self.poly_order, self.include_sine)
        dz = Theta @ (self.coefficients * self.coefficient_mask)
        return z + dz * dt

    def loss(self,
             z_curr: torch.Tensor,
             z_next: torch.Tensor) -> torch.Tensor:
        z_pred = self.predict_next(z_curr)
        return F.mse_loss(z_pred, z_next)

    def print_equations(self,
                        threshold: float = 1e-6,
                        include_constant: bool = True) -> None:
        """
        Print discovered SINDy equations d z_i/dt = f_i(z).
        """
        names = _build_sindy_feature_names(
            self.latent_dim, self.poly_order, self.include_sine, include_constant
        )
        Xi = self.coefficients.detach().cpu().numpy()
        for i in range(self.latent_dim):
            terms = []
            for coef, name in zip(Xi[:, i], names):
                if abs(coef) > threshold:
                    terms.append(f"{coef:.6g}*{name}")
            rhs = " + ".join(terms) if terms else "0"
            print(f"dz{i}/dt = {rhs}")

    def simulate(self,
                 z0: torch.Tensor,
                 n_steps: int,
                 dt: Optional[float] = None) -> torch.Tensor:
        """
        Forward-simulate latent dynamics from initial z0 for n_steps.
        Returns (n_steps+1, latent_dim).
        """
        dt = dt if dt is not None else self.dt
        zs = [z0]
        z = z0
        for _ in range(n_steps):
            z = self.predict_next(z, dt)
            zs.append(z)
        return torch.stack(zs, dim=0)
