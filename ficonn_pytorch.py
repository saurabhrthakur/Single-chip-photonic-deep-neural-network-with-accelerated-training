import torch
import torch.nn as nn
import numpy as np

class CmxuLayer(nn.Module):
    """
    A PyTorch module for a Coherent Matrix-vector Multiplication Unit (CMXU).
    This represents a single unitary transformation layer in the ONN.
    It has 36 trainable parameters for a 6x6 unitary matrix, composed of
    15 MZIs (30 phases) and 6 output phase shifters.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        
        # An N-port interferometer requires N*(N-1)/2 MZIs.
        # Each MZI has 2 phase parameters (theta, phi).
        num_mzis = n_channels * (n_channels - 1) // 2
        
        # Initialize the MZI phase parameters
        self.mzi_phases = nn.Parameter(torch.rand(num_mzis * 2) * 2 * np.pi)
        
        # Initialize the output phase shifter parameters
        self.output_phases = nn.Parameter(torch.rand(n_channels) * 2 * np.pi)

    def forward(self, complex_field: torch.Tensor) -> torch.Tensor:
        """
        Constructs the unitary matrix from phases and applies it to the complex input field.
        """
        unitary_matrix = self.build_unitary()
        return torch.matmul(complex_field.T, unitary_matrix).T

    def build_unitary(self) -> torch.Tensor:
        """
        Constructs the full unitary matrix from the MZI and output phase parameters.
        """
        U = torch.eye(self.n_channels, dtype=torch.cfloat)
        mzi_params_index = 0
        
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                # Create a transformation matrix for the MZI acting on channels i and j
                T_ij = torch.eye(self.n_channels, dtype=torch.cfloat)
                
                # Get the theta and phi for the current MZI
                theta = self.mzi_phases[mzi_params_index]
                phi = self.mzi_phases[mzi_params_index + 1]
                mzi_params_index += 2

                # Construct the 2x2 MZI matrix
                mzi_matrix = torch.tensor([
                    [torch.exp(1j * phi) * torch.cos(theta), torch.sin(theta)],
                    [-torch.exp(1j * phi) * torch.sin(theta), torch.cos(theta)]
                ], dtype=torch.cfloat)
                
                # Place the 2x2 MZI matrix into the full transformation matrix
                T_ij[i, i] = mzi_matrix[0, 0]
                T_ij[i, j] = mzi_matrix[0, 1]
                T_ij[j, i] = mzi_matrix[1, 0]
                T_ij[j, j] = mzi_matrix[1, 1]
                
                # Apply this transformation to the overall unitary matrix
                U = torch.matmul(T_ij, U)

        # Apply the final output phase shifters
        output_phase_matrix = torch.diag(torch.exp(1j * self.output_phases))
        U = torch.matmul(output_phase_matrix, U)
        
        return U

class NofuLayer(nn.Module):
    """
    A PyTorch module for a Nonlinear Optical Function Unit (NOFU).
    This represents the activation function in the ONN.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        # In a real hardware model, these would be trainable parameters.
        # Here, we can model a simple nonlinear effect.
        self.gamma = nn.Parameter(torch.ones(n_channels))

    def forward(self, complex_field: torch.Tensor) -> torch.Tensor:
        """
        Applies a nonlinear phase shift based on the intensity of the complex field.
        """
        intensity = torch.abs(complex_field)**2
        nonlinear_phase_shift = torch.exp(1j * self.gamma * intensity)
        return complex_field * nonlinear_phase_shift


class FiconnModel(nn.Module):
    """
    The complete FICONN model implemented in PyTorch.
    This model stacks CMXU and NOFU layers to form a deep optical neural network.
    For the digital benchmark, this is a standard feedforward neural network.
    """
    def __init__(self, n_channels: int = 6, n_layers: int = 3, n_classes: int = 10, 
                 use_tanh_activation: bool = True, noise_level: float = 0.0):
        super().__init__()
        
        self.noise_level = noise_level

        layers = []
        in_features = n_channels
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_channels))
            if use_tanh_activation:
                layers.append(nn.Tanh())
            in_features = n_channels
        
        self.layers = nn.Sequential(*layers)
        self.output_classifier = nn.Linear(n_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the full forward pass of the FICONN model.
        """
        # Pass through the main layers of the network
        x = self.layers(x)
        
        # Add noise during training to regularize the model
        if self.training and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
            
        # The final classifier
        output = self.output_classifier(x)
        return output
