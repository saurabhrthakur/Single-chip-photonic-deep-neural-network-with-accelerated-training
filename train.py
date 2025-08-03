import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR

# Import from our new modular files
from vowel_dataset import get_vowel_data
from ficonn_pytorch import FiconnModel
from ficonn_core import create_crosstalk_matrix, train_ficonn
from ficonn_dashboard import print_performance_dashboard

# =============================================================================
# SECTION 2: IN-SITU FICONN TRAINING (HARDWARE-AWARE)
# =============================================================================
def run_in_situ_training():
    """
    Orchestrates the full in-situ training pipeline.
    This version is configured for a diagnostic run on an IDEAL, NOISELESS model.
    """
    print("--- Starting In-Situ Training Pipeline (DIAGNOSTIC MODE: IDEAL HARDWARE) ---")
    
    # 1. Load data and determine model dimensions
    X_train, X_test, y_train, y_test, _, class_names = get_vowel_data()
    n_classes = len(class_names)
    n_channels = X_train.shape[1] # Should be 6
    n_layers = 3 # Fixed as per paper
    
    # Calculate parameters based on the paper's formula for a 6x6x6 network
    n_neuron = n_channels
    n_cmxu_params_per_layer = n_neuron * n_neuron # 36
    n_nofu_params_per_layer = 2 * n_neuron      # 12
    
    n_params = (n_layers * n_cmxu_params_per_layer) + ((n_layers - 1) * n_nofu_params_per_layer)
    # n_params = (3 * 36) + (2 * 12) = 108 + 24 = 132
    print(f"Calculated number of parameters: {n_params}")
    
    # Initialize model parameters
    initial_theta = np.random.uniform(0, 2 * np.pi, size=n_params)

    # 2. Define the Digital Twin (IDEAL, NOISELESS VERSION FOR DIAGNOSTICS)
    print("--- Defining IDEAL Digital Twin (Noiseless) ---")
    ideal_hardware_params = {
        # For the ideal test, we use an identity matrix = NO CROSSTALK
        "crosstalk_matrix": np.eye(n_params) 
    }
    
    # 3. Run the derivative-free, hardware-aware training (full-batch)
    final_theta, final_test_acc = train_ficonn(
        initial_theta=initial_theta,
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test,
        y_test=y_test,
        noisy_hardware_params=ideal_hardware_params, # Using the ideal params
        n_epochs=1000,               # Longer run to ensure it has time to learn
        learning_rate=0.002,         # Using the paper's specified LR
        delta=0.05
    )
    
    print("--- In-Situ Training Pipeline Finished ---")
    
    # 4. Display final performance dashboard
    print_performance_dashboard(final_test_acc)


if __name__ == '__main__':
    run_in_situ_training()
