import numpy as np
from tqdm import tqdm
from scipy.fft import fft, ifft

# ==============================================================================
# SECTION 1: CORE ONN PHYSICS & FORWARD PASS
# ==============================================================================

def construct_unitary_fourier_interlaced(params, n):
    """
    Constructs an n x n unitary matrix using the Fourier-interlaced
    architecture: alternating diagonal phase masks and fixed DFT matrices.
    This is a physically realistic and robust parameterization.
    """
    if len(params) != n * n:
        raise ValueError(f"Incorrect number of parameters. Expected {n*n}, got {len(params)} for a {n}x{n} matrix.")

    # Fixed mixing layer: Discrete Fourier Transform matrix
    F = fft(np.eye(n)) / np.sqrt(n)

    # Start with identity
    U = np.eye(n, dtype=np.complex128)
    
    # Reshape parameters for easier handling
    phase_params = params.reshape((n, n))
    
    # Cascade of N layers of phase masks and DFTs
    for i in range(n):
        D = np.diag(np.exp(1j * phase_params[i, :]))
        U = F @ D @ U
        
    return U

def vector_to_params(theta, n_channels):
    """
    Converts a flat vector of 132 parameters into a structured dictionary 
    for a 3-layer, 6x6x6 ONN using the Fourier-interlaced CMXU model.
    """
    params = []
    current_pos = 0
    n_cmxu_params = n_channels * n_channels # 36 for a 6x6 CMXU
    n_nofu_params = n_channels * 2         # 12 for 6 NOFUs (2 params each)
    
    # Layer 1
    params.append({
        'cmxu_params': theta[current_pos : current_pos + n_cmxu_params],
        'nofu_params': theta[current_pos + n_cmxu_params : current_pos + n_cmxu_params + n_nofu_params]
    })
    current_pos += n_cmxu_params + n_nofu_params

    # Layer 2
    params.append({
        'cmxu_params': theta[current_pos : current_pos + n_cmxu_params],
        'nofu_params': theta[current_pos + n_cmxu_params : current_pos + n_cmxu_params + n_nofu_params]
    })
    current_pos += n_cmxu_params + n_nofu_params

    # Layer 3 (Output) - CMXU only
    params.append({
        'cmxu_params': theta[current_pos : current_pos + n_cmxu_params]
    })
    
    return params

def _apply_activation(vector, nofu_params):
    """
    Applies a programmable non-linearity, simulating the NOFU.
    """
    split = len(nofu_params) // 2
    magnitude_params = nofu_params[:split]
    phase_params = nofu_params[split:]
    
    magnitude = np.tanh(magnitude_params * np.abs(vector))
    phase = np.angle(vector) + phase_params
    return magnitude * np.exp(1j * phase)

def onn_forward_complex_noisy(theta_intended, crosstalk_matrix, x_fields, n_channels):
    """
    Simulates the forward pass of the 6x6x6 FICONN, returning the final complex vector.
    """
    theta_actual = crosstalk_matrix @ theta_intended
    params = vector_to_params(theta_actual, n_channels)
    a = np.asarray(x_fields, dtype=np.complex128)

    for i in range(3): # 3 layers
        layer_params = params[i]
        
        # Construct the unitary matrix for the CMXU
        U_noisy = construct_unitary_fourier_interlaced(layer_params['cmxu_params'], n=n_channels)
        a = U_noisy @ a
        
        # Apply non-linearity (NOFU) if it's not the last layer
        if 'nofu_params' in layer_params:
            a = _apply_activation(a, layer_params['nofu_params'])
            
    return a

def cross_entropy_loss(logits, target_idx):
    """
    Computes the cross-entropy loss from a vector of real-valued logits.
    """
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    loss = -np.log(probs[target_idx] + 1e-9)
    return loss, probs

def calculate_total_loss_and_accuracy(theta, X, y, noisy_hardware_params, n_channels, n_classes):
    """
    Calculates total loss and accuracy using the corrected 6x6x6 architecture
    and I/Q readout to get 12 output logits.
    """
    total_loss = 0
    correct_predictions = 0
    
    for i in range(len(X)):
        input_vector = np.exp(1j * np.pi * X[i])
        
        final_complex_vector = onn_forward_complex_noisy(
            theta_intended=theta,
            crosstalk_matrix=noisy_hardware_params["crosstalk_matrix"],
            x_fields=input_vector,
            n_channels=n_channels
        )
        
        if len(final_complex_vector) * 2 != n_classes:
             raise ValueError("Mismatch between readout vector and number of classes!")
        logits = np.concatenate([np.real(final_complex_vector), np.imag(final_complex_vector)])
        
        loss, probs = cross_entropy_loss(logits, y[i])
        total_loss += loss
        
        if np.argmax(probs) == y[i]:
            correct_predictions += 1
            
    avg_loss = total_loss / len(X)
    accuracy = (correct_predictions / len(X)) * 100
    return avg_loss, accuracy

# ==============================================================================
# SECTION 2: IN-SITU TRAINING
# ==============================================================================

def create_crosstalk_matrix(n_params, crosstalk_factor=0.1):
    """Creates a crosstalk matrix to simulate thermal leakage."""
    M = np.eye(n_params)
    for i in range(n_params):
        for j in range(n_params):
            if i != j:
                M[i, j] = crosstalk_factor / (abs(i - j) + 1)
    return M

def train_ficonn(initial_theta, X_train, y_train, X_test, y_test, noisy_hardware_params, 
                 n_epochs, learning_rate, delta):
    """
    Trains the FICONN model using the derivative-free, full-batch in-situ algorithm.
    """
    print("Starting FICONN in-situ training (full-batch method)...")
    
    theta = initial_theta.copy()
    n_params = len(theta)
    n_channels = X_train.shape[1]
    n_classes = len(np.unique(y_train))

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training Epochs"):
        delta_vector = np.random.choice([-delta, delta], size=n_params)
        
        loss_plus, _ = calculate_total_loss_and_accuracy(
            theta + delta_vector, X_train, y_train, noisy_hardware_params, n_channels, n_classes
        )
        
        loss_minus, _ = calculate_total_loss_and_accuracy(
            theta - delta_vector, X_train, y_train, noisy_hardware_params, n_channels, n_classes
        )

        theta -= learning_rate * (loss_plus - loss_minus) * delta_vector
        
        if epoch % 5 == 0 or epoch == n_epochs:
            train_loss, train_acc = calculate_total_loss_and_accuracy(theta, X_train, y_train, noisy_hardware_params, n_channels, n_classes)
            test_loss, test_acc = calculate_total_loss_and_accuracy(theta, X_test, y_test, noisy_hardware_params, n_channels, n_classes)
            print(f"\n--- Epoch {epoch}/{n_epochs} Report ---")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

    print("\nFinished FICONN training.")
    # Final evaluation
    final_train_loss, final_train_acc = calculate_total_loss_and_accuracy(theta, X_train, y_train, noisy_hardware_params, n_channels, n_classes)
    final_test_loss, final_test_acc = calculate_total_loss_and_accuracy(theta, X_test, y_test, noisy_hardware_params, n_channels, n_classes)
    print(f"\n--- Final Performance ---")
    print(f"Train Loss: {final_train_loss:.4f}, Train Acc: {final_train_acc:.2f}%")
    print(f"Test Loss:  {final_test_loss:.4f}, Test Acc:  {final_test_acc:.2f}%")

    return theta
