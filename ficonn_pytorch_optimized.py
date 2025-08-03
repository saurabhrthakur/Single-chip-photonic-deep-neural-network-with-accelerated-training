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
        
        # Better initialization: Xavier/Glorot initialization for better convergence
        self.mzi_phases = nn.Parameter(torch.randn(num_mzis * 2) * 0.1)
        self.output_phases = nn.Parameter(torch.randn(n_channels) * 0.1)
        
        # Add custom dropout for complex tensors
        self.dropout = ComplexDropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Construct the unitary matrix from the current phase parameters
        U = self.construct_unitary()
        
        # Apply the unitary transformation
        output = torch.matmul(U, x.unsqueeze(-1)).squeeze(-1)
        
        # Apply dropout to the output
        output = self.dropout(output)
        
        return output

    def construct_unitary(self) -> torch.Tensor:
        """
        Constructs the N x N unitary matrix from the phase parameters.
        """
        U = torch.eye(self.n_channels, dtype=torch.cfloat, device=self.mzi_phases.device)
        phase_idx = 0
        
        for i in range(self.n_channels):
            for j in range(i + 1, self.n_channels):
                theta = self.mzi_phases[phase_idx]
                phi = self.mzi_phases[phase_idx + 1]
                phase_idx += 2
                
                c = np.sqrt(0.5)
                s = np.sqrt(0.5)
                
                T_ij = torch.eye(self.n_channels, dtype=torch.cfloat, device=U.device)
                
                T_ij[i, i] = c
                T_ij[i, j] = s * torch.exp(1j * phi)
                T_ij[j, i] = s * torch.exp(1j * theta)
                T_ij[j, j] = -c * torch.exp(1j * (theta + phi))

                U = T_ij @ U
                
        # Apply the final output phase shifters
        D = torch.diag(torch.exp(1j * self.output_phases))
        U_final = D @ U
        return U_final


class OptimizedNofuLayer(nn.Module):
    """
    An optimized NOFU layer with better activation functions and regularization.
    """
    def __init__(self, n_channels: int, photodiode_responsivity: float = 0.8, 
                 sensitivity: float = 1e-3, fwhm: float = 0.1, dropout_rate: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        
        # Better initialization
        self.beta_param = nn.Parameter(torch.randn(n_channels) * 0.1)
        self.initial_detuning = nn.Parameter(torch.randn(n_channels) * 0.1)
        
        # Constants
        self.photodiode_responsivity = photodiode_responsivity
        self.sensitivity = sensitivity
        self.fwhm = fwhm
        
        # Add custom dropout for complex tensors
        self.dropout = ComplexDropout(dropout_rate)

    def _current_to_detuning(self, current: torch.Tensor) -> torch.Tensor:
        return self.sensitivity * current

    def _lorentzian_transmission_complex(self, detuning: torch.Tensor) -> torch.Tensor:
        # Complex Lorentzian transmission function
        gamma = self.fwhm / 2
        transmission = gamma / (gamma + 1j * detuning)
        return transmission

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Constrain beta to be between 0 and 1 using sigmoid
        beta = torch.sigmoid(self.beta_param)
        beta = beta.unsqueeze(0)
        initial_detuning = self.initial_detuning.unsqueeze(0)

        # Calculate power and resulting detuning
        input_power = torch.abs(x)**2
        tapped_power = beta * input_power
        photocurrent = self.photodiode_responsivity * tapped_power
        detuning_change = self._current_to_detuning(photocurrent)
        final_detuning = initial_detuning + detuning_change
        
        # The remaining field passes through the microring
        sqrt_1_minus_beta = torch.sqrt(torch.relu(1 - beta))
        field_to_microring = sqrt_1_minus_beta * x

        # Get the complex transmission of the microring
        transmission = self._lorentzian_transmission_complex(final_detuning)

        # Apply the transmission to the field
        output_field = field_to_microring * transmission
        
        # Apply dropout
        output_field = self.dropout(output_field)
        
        return output_field


class OptimizedFiconnModel(nn.Module):
    """
    An optimized FICONN model with deeper architecture, better regularization, and improved training.
    """
    def __init__(self, n_channels: int = 6, n_layers: int = 5, n_classes: int = 12, 
                 use_tanh_activation: bool = True, dropout_rate: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.use_tanh_activation = use_tanh_activation
        self.dropout_rate = dropout_rate

        # Deeper architecture with more layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(OptimizedCmxuLayer(n_channels, dropout_rate))
            
            # Add activation after each CMXU layer except the last
            if i < n_layers - 1:
                if self.use_tanh_activation:
                    # Use a more sophisticated activation function
                    self.layers.append(nn.Sequential(
                        nn.Tanh(),
                        nn.Dropout(dropout_rate)
                    ))
                else:
                    self.layers.append(OptimizedNofuLayer(n_channels, dropout_rate=dropout_rate))
        
        # Enhanced output classifier with multiple layers
        self.output_classifier = nn.Sequential(
            nn.Linear(n_channels, n_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_channels * 2, n_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_channels, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input encoding
        complex_field = x.to(torch.cfloat)

        # Layered processing
        for layer in self.layers:
            if isinstance(layer, OptimizedCmxuLayer):
                complex_field = layer(complex_field)
            elif isinstance(layer, OptimizedNofuLayer):
                complex_field = layer(complex_field)
            elif isinstance(layer, nn.Sequential):
                # Apply tanh to real and imaginary parts separately
                real_part = layer(complex_field.real)
                imag_part = layer(complex_field.imag)
                complex_field = torch.complex(real_part, imag_part)
            
        # Coherent readout
        output_power = torch.abs(complex_field)**2
        
        # Classification
        logits = self.output_classifier(output_power)
        
        return logits


class OptimizedTrainingConfig:
    """
    Configuration class for optimized training parameters.
    """
    def __init__(self):
        self.learning_rate = 0.001
        self.batch_size = 64
        self.epochs = 200
        self.weight_decay = 1e-4
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        self.early_stopping_patience = 20
        self.dropout_rate = 0.2
        self.n_layers = 5 