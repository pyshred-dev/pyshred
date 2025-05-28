from ..models._sindy import sindy_library_torch, e_sindy_library_torch
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SINDy(torch.nn.Module):
    def __init__(self, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h, dt):
        library_Theta = sindy_library_torch(h, self.latent_dim, self.poly_order, self.include_sine)
        h = h + library_Theta @ (self.coefficients * self.coefficient_mask) * dt
        return h
    
    def thresholding(self, threshold):
        self.coefficient_mask = torch.abs(self.coefficients) > threshold
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.library_dim, self.latent_dim, requires_grad=False).to(device)   

class E_SINDy(torch.nn.Module):
    def __init__(self, num_replicates, latent_dim, library_dim, poly_order, include_sine):
        super().__init__()
        self.num_replicates = num_replicates
        self.latent_dim = latent_dim
        self.poly_order = poly_order
        self.include_sine = include_sine
        self.library_dim = library_dim
        self.coefficients = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=True)
        torch.nn.init.normal_(self.coefficients, mean=0.0, std=0.001)
        self.coefficient_mask = torch.ones(num_replicates, library_dim, latent_dim, requires_grad=False).to(device)
        self.coefficients = torch.nn.Parameter(self.coefficients)

    def forward(self, h_replicates, dt):
        num_data, num_replicates, latent_dim = h_replicates.shape
        h_replicates = h_replicates.reshape(num_data * num_replicates, latent_dim)
        library_Thetas = e_sindy_library_torch(h_replicates, self.latent_dim, self.poly_order, self.include_sine)
        library_Thetas = library_Thetas.reshape(num_data, num_replicates, self.library_dim)
        h_replicates = h_replicates.reshape(num_data, num_replicates, latent_dim)
        h_replicates = h_replicates + torch.einsum('ijk,jkl->ijl', library_Thetas, (self.coefficients * self.coefficient_mask)) * dt
        return h_replicates
    
    def thresholding(self, threshold, base_threshold=0):
        threshold_tensor = torch.full_like(self.coefficients, threshold)
        for i in range(self.num_replicates):
            threshold_tensor[i] = threshold_tensor[i] * 10**(0.2 * i - 1) + base_threshold
        self.coefficient_mask = torch.abs(self.coefficients) > threshold_tensor
        self.coefficients.data = self.coefficient_mask * self.coefficients.data
        
    def add_noise(self, noise=0.1):
        self.coefficients.data += torch.randn_like(self.coefficients.data) * noise
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)   
        
    def recenter(self):
        self.coefficients.data = torch.randn_like(self.coefficients.data) * 0.0
        self.coefficient_mask = torch.ones(self.num_replicates, self.library_dim, self.latent_dim, requires_grad=False).to(device)  