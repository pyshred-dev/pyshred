from ..models._sindy import sindy_library_torch, e_sindy_library_torch
import torch
from .device import get_device

class SINDy(torch.nn.Module):
    """
    Sparse Identification of Nonlinear Dynamics (SINDy) module.
    
    A PyTorch implementation of SINDy for discovering sparse dynamical systems
    from data. Uses polynomial and optional sine libraries to represent
    the dynamics with sparse coefficients.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space.
    library_dim : int
        Dimension of the feature library.
    poly_order : int
        Order of polynomial terms in the library.
    include_sine : bool
        Whether to include sine functions in the library.

    Attributes
    ----------
    coefficients : torch.nn.Parameter
        Learnable coefficients of the SINDy model.
    coefficient_mask : torch.Tensor
        Binary mask for enforcing sparsity.
    """
    
    def __init__(self, latent_dim, library_dim, poly_order, include_sine):
        """
        Initialize the SINDy module.

        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        library_dim : int
            Dimension of the feature library.
        poly_order : int
            Order of polynomial terms in the library.
        include_sine : bool
            Whether to include sine functions in the library.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        device = get_device()
        self.coefficient_mask = torch.ones(library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h, dt):
        """
        Forward pass through the SINDy dynamics.

        Parameters
        ----------
        h : torch.Tensor
            Current latent state.
        dt : float
            Time step size.

        Returns
        -------
        torch.Tensor
            Updated latent state after one time step.
        """
        library_Theta = sindy_library_torch(h, self.latent_dim, self.poly_order, self.include_sine)
        h = h + library_Theta @ (self.coefficients * self.coefficient_mask) * dt
        return h
    
    def thresholding(self, threshold):
        """
        Apply thresholding to enforce sparsity in coefficients.

        Parameters
        ----------
        threshold : float
            Threshold value below which coefficients are set to zero.
        """
        self.coefficient_mask = torch.abs(self.coefficients) > threshold
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        """
        Add noise to coefficients and reset mask.

        Parameters
        ----------
        noise : float, optional
            Standard deviation of noise to add. Defaults to 0.1.
        """
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        device = get_device()
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)
        
    def recenter(self):
        """
        Reset coefficients to zero and mask to ones.
        """
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        device = get_device()
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)

class E_SINDy(torch.nn.Module):
    """
    Ensemble Sparse Identification of Nonlinear Dynamics (E-SINDy) module.
    
    An ensemble version of SINDy that uses multiple replicate models
    to improve robustness and handle noise in the data.

    Parameters
    ----------
    num_replicates : int
        Number of SINDy models in the ensemble.
    latent_dim : int
        Dimension of the latent space.
    library_dim : int
        Dimension of the feature library.
    poly_order : int
        Order of polynomial terms in the library.
    include_sine : bool
        Whether to include sine functions in the library.

    Attributes
    ----------
    coefficients : torch.nn.Parameter
        Learnable coefficients for all ensemble members.
    coefficient_mask : torch.Tensor
        Binary masks for enforcing sparsity in each ensemble member.
    """
    
    def __init__(self, num_replicates, latent_dim, library_dim, poly_order, include_sine):
        """
        Initialize the E-SINDy ensemble module.

        Parameters
        ----------
        num_replicates : int
            Number of SINDy models in the ensemble.
        latent_dim : int
            Dimension of the latent space.
        library_dim : int
            Dimension of the feature library.
        poly_order : int
            Order of polynomial terms in the library.
        include_sine : bool
            Whether to include sine functions in the library.
        """
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        device = get_device()
        self.coefficient_mask = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h_replicates, dt):
        """
        Forward pass through the E-SINDy ensemble dynamics.

        Parameters
        ----------
        h_replicates : torch.Tensor
            Current latent states for all replicates, shape (num_data, num_replicates, latent_dim).
        dt : float
            Time step size.

        Returns
        -------
        torch.Tensor
            Updated latent states for all replicates after one time step.
        """
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(h_replicates, self.latent_dim, self.poly_order, self.include_sine)
        library_Thetas = library_Thetas.reshape(num_data, num_replicates, self.library_dim)
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = h_replicates + torch.einsum('ijk,jkl->ijl', library_Thetas, (self.coefficients * self.coefficient_mask)) * dt
        return h_replicates
    
    def thresholding(self, threshold, base_threshold=0):
        """
        Apply varying thresholds to different ensemble members for sparsity.

        Parameters
        ----------
        threshold : float
            Base threshold value.
        base_threshold : float, optional
            Minimum threshold to add to scaled threshold. Defaults to 0.
        """
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        """
        Add noise to coefficients and reset masks.

        Parameters
        ----------
        noise : float, optional
            Standard deviation of noise to add. Defaults to 0.1.
        """
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        device = get_device()
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)   
        
    def recenter(self):
        """
        Reset all coefficients to zero and masks to ones.
        """
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        device = get_device()
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)  