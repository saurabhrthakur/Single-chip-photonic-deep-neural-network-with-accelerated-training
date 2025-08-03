import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from ficonn_core import onn_forward_complex_noisy

# ==============================================================================
# SECTION 1: DIGITAL TWIN FITTING
# ==============================================================================

def fit_digital_twin_lbfgs(initial_params, fixed_crosstalk_matrix, X_train, y_train, n_params, input_dim, max_iter=10, batch_size=64):
    """
    Fits a digital twin model to training data using L-BFGS-B optimization.
    This version is optimized to only learn scalar parameters for speed.
    """
    # Extract initial values for the 3 parameters we are learning
    initial_loss_mean, initial_loss_std = initial_params["insertion_loss"]
    initial_bs_error = initial_params["beamsplitter_error"]
    
    initial_flat_params = np.array([initial_loss_mean, initial_loss_std, initial_bs_error])
    
    def loss_function(flat_params):
        loss_mean_db, loss_std_db, bs_error = flat_params
        total_loss = 0.0
        
        indices = np.random.choice(len(X_train), batch_size, replace=False)
        X_batch, y_batch = X_train[indices], y_train[indices]
        
        for i in range(batch_size):
            phases = X_batch[i][:n_params]
            input_real = X_batch[i][n_params:n_params+input_dim]
            input_imag = X_batch[i][n_params+input_dim:]
            input_vector = input_real + 1j * input_imag
            
            twin_output = onn_forward_complex_noisy(
                theta_intended=phases,
                crosstalk_matrix=fixed_crosstalk_matrix,
                x_fields=input_vector,
                lo_field=1.0 + 0j,
                n_layers=3,
                n_channels=input_dim,
                loss_mean_db=loss_mean_db,
                loss_std_db=loss_std_db,
                beamsplitter_error_std=bs_error
            )
            
            ground_truth = y_batch[i]
            mse = np.mean(np.abs(twin_output - ground_truth)**2)
            if np.isnan(mse):
                return 1e12
            total_loss += mse
        
        avg_loss = total_loss / batch_size
        if np.random.rand() < 0.1:
             print(f"Current loss: {avg_loss:.6f}")
        return avg_loss

    bounds = [
        (0.0, None),
        (0.0, None),
        (-0.1, 0.1),
    ]
    
    print(f"Starting L-BFGS-B optimization with {max_iter} max iterations...")
    history = []
    
    def callback(x):
        history.append(loss_function(x))
    
    result = minimize(
        loss_function,
        initial_flat_params,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'disp': True},
        callback=callback
    )
    
    trained_loss_mean, trained_loss_std, trained_bs_error = result.x
    
    trained_params = {
        "crosstalk_matrix": fixed_crosstalk_matrix,
        "insertion_loss": (trained_loss_mean, trained_loss_std),
        "beamsplitter_error": trained_bs_error
    }
    
    print(f"Optimization complete. Final loss: {result.fun:.6f}")
    
    return trained_params, history
