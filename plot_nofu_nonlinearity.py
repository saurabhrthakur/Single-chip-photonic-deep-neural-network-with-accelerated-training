#!/usr/bin/env python3
"""
Script to visualize the NOFU nonlinearity in response to different input powers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt
from ficonn_core import nofu_activation_with_bias, verify_nofu_physics, model_passive_ring_response

def plot_nofu_response_curve(beta=0.2, delta_lambda=0.1, V_B=0.8):
    # Generate logarithmically spaced power levels (more points at low power)
    powers = np.logspace(-6, -1, 100)  # 1µW to 100mW
    
    # Calculate transmission at each power level
    transmissions = []
    phases = []
    
    for power in powers:
        # Create complex input field
        input_field = np.sqrt(power) + 0j
        
        # Calculate NOFU response
        output_field = nofu_activation_with_bias(input_field, beta, delta_lambda, V_B)
        
        # Calculate transmission and phase
        transmission = np.abs(output_field)**2 / power
        phase = np.angle(output_field)
        
        transmissions.append(transmission)
        phases.append(phase)
    
    # Convert to arrays
    transmissions = np.array(transmissions)
    phases = np.array(phases)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot transmission vs power
    ax1.semilogx(powers, transmissions, 'b-', linewidth=2)
    ax1.set_xlabel('Input Power (W)')
    ax1.set_ylabel('Transmission |Eout|²/|Ein|²')
    ax1.set_title(f'NOFU Transmission (β={beta}, Δλ={delta_lambda}nm, V_B={V_B}V)')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot phase vs power
    ax2.semilogx(powers, phases, 'r-', linewidth=2)
    ax2.set_xlabel('Input Power (W)')
    ax2.set_ylabel('Phase (rad)')
    ax2.set_title('NOFU Phase Response')
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('nofu_nonlinearity_response.png', dpi=300)
    plt.show()
    
    # Return nonlinearity check
    nonlinear = np.std(transmissions) > 0.01
    return nonlinear

def compare_multiple_parameter_sets():
    """Compare multiple NOFU parameter configurations"""
    # Define parameter sets from the paper (β, Δλ)
    param_sets = [
        (0.2, 0.1),
        (0.4, 0.1),
        (0.4, 0.2),
        (0.2, 0.2),
        (0.2, 0.15),
        (0.4, 0.25)
    ]
    
    # Generate logarithmically spaced power levels
    powers = np.logspace(-6, -1, 100)  # 1µW to 100mW
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Colors for different curves
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    # Calculate and plot transmission for each parameter set
    for i, (beta, delta_lambda) in enumerate(param_sets):
        transmissions = []
        
        for power in powers:
            input_field = np.sqrt(power) + 0j
            output_field = nofu_activation_with_bias(input_field, beta, delta_lambda, V_B=0.8)
            transmission = np.abs(output_field)**2 / power
            transmissions.append(transmission)
        
        plt.semilogx(powers, transmissions, f'{colors[i]}-', 
                    linewidth=2, label=f'β={beta}, Δλ={delta_lambda}nm')
    
    plt.xlabel('Input Power (W)')
    plt.ylabel('Transmission |Eout|²/|Ein|²')
    plt.title('NOFU Transmission for Different Parameter Sets')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('nofu_parameter_comparison.png', dpi=300)
    plt.show()

def plot_nofu_response_figS3a():
    """
    Plot NOFU wavelength response vs current (Figure S3a)
    Shows transmission vs wavelength curves for different injected currents,
    demonstrating the transition from overcoupled to undercoupled regime
    """
    # Define wavelength range (nm) - similar to Fig S3a
    wavelengths = np.linspace(1563.0, 1565.0, 500)
    
    # Define current values (µA) as shown in Fig S3a
    currents_uA = [0, 74, 156, 229, 309, 382, 452]
    
    # Define colors for different current values
    colors = ['blue', 'red', 'orange', 'teal', 'green', 'gold', 'purple']
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Center wavelength for reference
    center_wavelength = 1564.0
    
    # Fixed parameters for NOFU model
    beta = 0.2  # Tapped power fraction
    V_B = 0.8   # Bias voltage (positive for injection mode)
    
    # Plot transmission for each current value
    for i, current_uA in enumerate(currents_uA):
        transmissions = []
        
        # Convert microamps to mA
        current_mA = current_uA / 1000
        
        for wl in wavelengths:
            # Calculate detuning from center wavelength
            delta_lambda = wl - center_wavelength
            
            # Create unit amplitude input field
            input_field = 1.0 + 0j
            
            # Use the NOFU activation function from ficonn_core
            # We'll simulate the current effect by adjusting the photocurrent
            # This is an approximation of how the NOFU would respond at different wavelengths
            
            # For simulation purposes, we'll create a modified version of nofu_activation_with_bias
            # that takes wavelength as input instead of using it directly
            
            # Create a simulated photocurrent based on the given current values
            simulated_I = current_mA
            
            # Call NOFU with the current wavelength detuning
            output_field = nofu_activation_with_bias(
                input_field, 
                beta=beta,
                delta_lambda_initial=delta_lambda,
                V_B=V_B,
                # Override the I_scale to directly use our current values
                I_scale=simulated_I * 5 if simulated_I > 0 else 0.001
            )
            
            # Calculate transmission in dB
            transmission = np.abs(output_field)**2
            trans_dB = 20 * np.log10(transmission) if transmission > 0 else -15
            transmissions.append(trans_dB)
        
        plt.plot(wavelengths, transmissions, color=colors[i], 
                 linewidth=2, label=f'I = {current_uA} µA')
    
    # Set plot properties
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (dB)')
    plt.title('NOFU Response vs. Current (Figure S3a)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Set y-axis limits similar to Fig S3a
    plt.ylim(-15, 0.5)
    
    # Save and show the figure
    plt.tight_layout()
    plt.savefig('nofu_response_figS3a.png', dpi=300)
    plt.show()

# In nofu_activation_testing.py, replace the old plotting function with this one.
# Make sure to import model_passive_ring_response from ficonn_core

def plot_nofu_response_figS3a_corrected():
    """
    Correctly plots the NOFU's passive wavelength response vs. an injected current,
    replicating the experiment shown in Figure S3a.
    """
    # Define wavelength range (nm) - similar to Fig S3a
    wavelengths = np.linspace(1563.0, 1565.0, 400)
    
    # Define current values (µA) as shown in Fig S3a
    currents_uA = [0, 74, 156, 229, 309, 382, 452]
    
    # Colors for different current values from a perceptually uniform colormap
    colors = plt.cm.viridis(np.linspace(0, 1, len(currents_uA)))
    
    plt.figure(figsize=(8, 6))
    
    center_wavelength = 1564.05  # Fine-tuned to match the plot center

    for i, current_uA in enumerate(currents_uA):
        transmissions_dB = []
        current_mA = current_uA / 1000.0
        
        for wl in wavelengths:
            delta_lambda = wl - center_wavelength
            
            # Call the NEW passive model function
            transmission_linear = model_passive_ring_response(
                delta_lambda=delta_lambda,
                injected_current_mA=current_mA
            )
            
            # Convert power transmission to dB for plotting
            trans_dB = 10 * np.log10(transmission_linear + 1e-9)
            transmissions_dB.append(trans_dB)
            
        plt.plot(wavelengths, transmissions_dB, color=colors[i], 
                 linewidth=3, label=f'I = {current_uA} µA')
    
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Transmission (dB)', fontsize=12)
    plt.title('Simulated NOFU Response vs. Current (Replicating Fig. S3a)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right')
    plt.ylim(-15, 0.5)
    plt.xlim(1563.0, 1565.0)
    
    plt.tight_layout()
    plt.savefig('nofu_response_figS3a_CORRECTED.png', dpi=300)
    plt.show()

# In your if __name__ == "__main__": block, call the new function:
# plot_nofu_response_figS3a_corrected()


# In nofu_activation_testing.py, replace the old S3b plotting function with this one.
# Make sure to import model_passive_ring_response from ficonn_core

def plot_nofu_resonance_figS3b_corrected():
    """
    Correctly plots and calibrates the NOFU's passive resonance at zero current,
    replicating the high-Q behavior shown in Figure S3b.
    """
    # 1. --- CORRECTED PHYSICAL PARAMETERS ---
    # To achieve a high Q of ~8300, the internal loss must be very low,
    # and kappa must be small to match it for critical coupling.
    CORRECTED_KAPPA = 0.05
    CORRECTED_INTRINSIC_A = 0.978

    # 2. --- SETUP SIMULATION ---
    center_wavelength = 1564.08  # Slightly tuned for perfect centering
    
    # The FSR of the device is ~0.8 nm. We plot a range smaller than this
    # to ensure we only see ONE resonance peak, just like in the paper.
    wavelength_span = 0.7
    wavelengths = np.linspace(center_wavelength - wavelength_span / 2,
                              center_wavelength + wavelength_span / 2, 800)
    
    # 3. --- RUN THE PASSIVE MODEL with Corrected Parameters ---
    transmissions_linear = []
    for wl in wavelengths:
        delta_lambda = wl - center_wavelength
        
        transmission = model_passive_ring_response(
            delta_lambda=delta_lambda,
            injected_current_mA=0.0,
            kappa=CORRECTED_KAPPA,                 # <-- USE CORRECTED VALUE
            intrinsic_a=CORRECTED_INTRINSIC_A      # <-- USE CORRECTED VALUE
        )
        transmissions_linear.append(transmission)
        
    transmissions_linear = np.array(transmissions_linear)
    transmissions_dB = 10 * np.log10(transmissions_linear + 1e-9)

    # 4. --- CALCULATE AND VERIFY THE Q FACTOR ---
    min_trans_dB = np.min(transmissions_dB)
    res_wavelength_idx = np.argmin(transmissions_dB)
    res_wavelength = wavelengths[res_wavelength_idx]
    half_max_dB = min_trans_dB / 2.0
    fwhm_indices = np.where(transmissions_dB <= half_max_dB)[0]
    
    Q_factor_calculated = 0
    if len(fwhm_indices) > 1:
        fwhm_nm = wavelengths[fwhm_indices[-1]] - wavelengths[fwhm_indices[0]]
        Q_factor_calculated = res_wavelength / fwhm_nm

    # 5. --- PLOT THE RESULTS ---
    plt.figure(figsize=(8, 6))
    plt.plot(wavelengths, transmissions_dB, 'r-', linewidth=3, label='Corrected Model Fit')
    
    plt.text(0.35, 0.7, f'Q ≈ {Q_factor_calculated:.0f}', transform=plt.gca().transAxes,
             fontsize=18, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Transmission (dB)', fontsize=12)
    plt.title('Calibrated NOFU Resonance (Replicating Fig. S3b)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.2)
    plt.legend()
    plt.ylim(-4.5, 0.5) # Match the paper's y-axis
    plt.xlim(wavelengths[0], wavelengths[-1]) # Use the new focused x-axis
    
    plt.tight_layout()
    plt.savefig('nofu_response_figS3b_CORRECTED.png', dpi=300)
    plt.show()

# --- HOW TO RUN ---
# In your if __name__ == "__main__": block, make sure you call this corrected function:
# plot_nofu_resonance_figS3b_corrected()

if __name__ == "__main__":
    print("Plotting NOFU nonlinear response curve...")
    
    # Run verification
    results = verify_nofu_physics(beta=0.2, initial_detuning=0.1)
    
    # Plot single response curve
    is_nonlinear = plot_nofu_response_curve(beta=0.2, delta_lambda=0.1)
    
    print(f"NOFU response is nonlinear: {is_nonlinear}")
    
    # Compare parameter sets from paper
    compare_multiple_parameter_sets()
    
    # Plot NOFU wavelength response (Fig S3a)
    plot_nofu_response_figS3a_corrected()
    plot_nofu_resonance_figS3b_corrected()

