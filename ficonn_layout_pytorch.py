#!/usr/bin/env python3
"""
FICONN LAYOUT - PYTORCH IMPLEMENTATION

This file implements the optimized PyTorch version of the FiCONN model
for optical neural networks.

Key components:
1. CMXU (Coherent Matrix-vector Multiplication Unit) - Unitary transformation
2. NOFU (Nonlinear Optical Function Unit) - Nonlinear activation
3. FiCONN Model - Full architecture with multiple layers
"""

import torch
import torch.nn as nn
import numpy as np

class ComplexDropout(nn.Module):
    """
    Custom dropout layer for complex tensors.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            # Create dropout mask for real and imaginary parts
            mask = torch.bernoulli(torch.ones_like(x.real) * (1 - self.p)) / (1 - self.p)
            return torch.complex(x.real * mask, x.imag * mask)
        return x

class OptimizedCmxuLayer(nn.Module):
    """
    An optimized CMXU layer with better initialization and regularization.
    """
    def __init__(self, n_channels: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        
        # An N-port interferometer requires N*(N-1)/2 MZIs.
        num_mzis = n_channels * (n_channels - 1) // 2
        
        # Initialize the MZI phase parameters with Glorot/Xavier initialization
        self.mzi_phases = nn.Parameter(torch.Tensor(num_mzis * 2))
        self._reset_parameters()
        
        # Initialize the output phase shifter parameters
        self.output_phases = nn.Parameter(torch.zeros(n_channels))
        
        # Dropout for regularization
        self.dropout = ComplexDropout(p=dropout_rate)

    def _reset_parameters(self):
        """Initialize parameters with improved distribution"""
        nn.init.uniform_(self.mzi_phases, 0, 2 * np.pi)

    def forward(self, complex_field: torch.Tensor) -> torch.Tensor:
        """
        Constructs the unitary matrix from phases and applies it to the complex input field.
        """
        unitary_matrix = self.build_unitary()
        
        # Apply the unitary transformation with dropout
        output = torch.matmul(complex_field, unitary_matrix)
        return self.dropout(output)
        
    def build_unitary(self) -> torch.Tensor:
        """
        Build the unitary matrix from the MZI phase parameters.
        This implements the Clements mesh architecture with improved numerical stability.
        """
        n = self.n_channels
        device = self.mzi_phases.device
        
        # Initialize identity matrix
        U = torch.eye(n, dtype=torch.complex64, device=device)
        
        # Build the unitary matrix layer by layer using the Clements scheme
        idx = 0
        for layer in range(n):
            for i in range(layer % 2, n-1, 2):
                j = i + 1
                
                # Get the current MZI parameters
                theta = self.mzi_phases[idx]
                phi = self.mzi_phases[idx + 1]
                idx += 2
                
                # Create the 2x2 unitary matrix for this MZI with improved stability
                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)
                exp_phi = torch.exp(1j * phi)
                exp_neg_phi = torch.exp(-1j * phi)
                
                T = torch.zeros((2, 2), dtype=torch.complex64, device=device)
                T[0, 0] = cos_theta * exp_phi
                T[0, 1] = sin_theta
                T[1, 0] = sin_theta
                T[1, 1] = -cos_theta * exp_neg_phi
                
                # Embed the 2x2 matrix into the full unitary
                Ti = torch.eye(n, dtype=torch.complex64, device=device)
                Ti[i:i+2, i:i+2] = T
                
                # Update the unitary matrix
                U = torch.matmul(U, Ti)
        
        # Apply the output phase shifters
        output_phases = self.output_phases
        diag = torch.exp(1j * output_phases)
        U = U * diag.unsqueeze(0)
        
        return U

class OptimizedNofuLayer(nn.Module):
    """
    Optimized Nonlinear Optical Function Unit (NOFU) layer.
    Implements a saturable absorption nonlinearity with improved stability.
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        
        # Two parameters per channel: alpha (absorption) and beta (saturation)
        # Initialize with better values
        self.alpha = nn.Parameter(torch.ones(n_channels) * 0.5)  # Stronger initial absorption
        self.beta = nn.Parameter(torch.ones(n_channels) * 0.2)   # Better saturation point
        
        # Add a small epsilon to prevent division by zero
        self.eps = 1e-8
        
    def forward(self, complex_field: torch.Tensor) -> torch.Tensor:
        """
        Apply the saturable absorption nonlinearity with improved numerical stability.
        """
        # Calculate the intensity (squared magnitude)
        intensity = torch.abs(complex_field)**2 + self.eps
        
        # Apply saturable absorption: T = exp(-alpha / (1 + beta * I))
        # Clamp values for stability
        alpha_clamped = torch.clamp(self.alpha, 0.0, 10.0)
        beta_clamped = torch.clamp(self.beta, 0.0, 10.0)
        
        transmission = torch.exp(-alpha_clamped.unsqueeze(0) / 
                                (1 + beta_clamped.unsqueeze(0) * intensity))
        
        # Apply the transmission to the complex field
        return complex_field * torch.sqrt(transmission)

class OptimizedTrainingConfig:
    """Configuration for optimized training"""
    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        self.dropout_rate = 0.1
        self.lr_step_size = 50
        self.lr_gamma = 0.9
        self.n_epochs = 500

class FiconnModel(nn.Module):
    """
    Optimized FiCONN model with improved initialization, regularization and stability.
    """
    def __init__(self, n_inputs: int, n_outputs: int, n_hidden: int = 6, n_layers: int = 3, 
                 dropout_rate: float = 0.1):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Create the layers
        self.layers = nn.ModuleList()
        
        # Input encoding layer (real to complex)
        self.input_phases = nn.Parameter(torch.zeros(n_inputs))
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(n_inputs)
        
        # Hidden layers
        for i in range(n_layers):
            # Add a CMXU layer
            self.layers.append(OptimizedCmxuLayer(n_hidden, dropout_rate))
            
            # Add a NOFU layer after the first two CMXU layers
            if i < 2:  # Only for first two layers as per paper
                self.layers.append(OptimizedNofuLayer(n_hidden))
        
        # Output decoding (complex to real)
        self.output_layer = nn.Linear(n_hidden * 2, n_outputs)
        
        # Output normalization
        self.output_norm = nn.BatchNorm1d(n_outputs)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the optimized FiCONN model.
        """
        batch_size = x.shape[0]
        
        # Apply input normalization
        if self.training:
            x = self.input_norm(x)
        
        # Encode real inputs to complex field
        # Use amplitude encoding: |ψ⟩ = x * exp(i * phase)
        amplitude = x
        phase = self.input_phases.unsqueeze(0).expand(batch_size, -1)
        complex_field = amplitude * torch.exp(1j * phase)
        
        # Pad or trim to match hidden dimension if needed
        if self.n_inputs != self.n_hidden:
            if self.n_inputs < self.n_hidden:
                # Pad with zeros
                padding = torch.zeros(batch_size, self.n_hidden - self.n_inputs, 
                                     dtype=complex_field.dtype, device=complex_field.device)
                complex_field = torch.cat([complex_field, padding], dim=1)
            else:
                # Trim
                complex_field = complex_field[:, :self.n_hidden]
        
        # Pass through the layers
        for layer in self.layers:
            complex_field = layer(complex_field)
        
        # Convert complex output to real
        # Concatenate real and imaginary parts
        real_output = torch.cat([complex_field.real, complex_field.imag], dim=1)
        
        # Final linear layer to get the outputs
        output = self.output_layer(real_output)
        
        # Apply output normalization
        if self.training:
            output = self.output_norm(output)
            
        return output

    def get_all_phases(self) -> torch.Tensor:
        """
        Extract all phase parameters from the model.
        Returns a flat tensor of all phases.
        """
        phases = []
        
        # Input phases
        phases.append(self.input_phases)
        
        # CMXU and NOFU phases
        for layer in self.layers:
            if isinstance(layer, OptimizedCmxuLayer):
                phases.append(layer.mzi_phases)
                phases.append(layer.output_phases)
            elif isinstance(layer, OptimizedNofuLayer):
                phases.append(layer.alpha)
                phases.append(layer.beta)
        
        # Concatenate all phases
        return torch.cat([p.flatten() for p in phases])
    
    def set_all_phases(self, phases: torch.Tensor) -> None:
        """
        Set all phase parameters from a flat tensor.
        """
        start_idx = 0
        
        # Input phases
        n_input_phases = self.n_inputs
        self.input_phases.data = phases[start_idx:start_idx+n_input_phases]
        start_idx += n_input_phases
        
        # CMXU and NOFU phases
        for layer in self.layers:
            if isinstance(layer, OptimizedCmxuLayer):
                n_mzi_phases = layer.mzi_phases.numel()
                layer.mzi_phases.data = phases[start_idx:start_idx+n_mzi_phases]
                start_idx += n_mzi_phases
                
                n_output_phases = layer.output_phases.numel()
                layer.output_phases.data = phases[start_idx:start_idx+n_output_phases]
                start_idx += n_output_phases
                
            elif isinstance(layer, OptimizedNofuLayer):
                n_alpha = layer.alpha.numel()
                layer.alpha.data = phases[start_idx:start_idx+n_alpha]
                start_idx += n_alpha
                
                n_beta = layer.beta.numel()
                layer.beta.data = phases[start_idx:start_idx+n_beta]
                start_idx += n_beta

# For backward compatibility - the optimized model is now the default
OptimizedFiconnModel = FiconnModel