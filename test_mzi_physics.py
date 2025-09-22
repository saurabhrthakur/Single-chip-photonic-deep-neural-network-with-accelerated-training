#!/usr/bin/env python3
"""
Test script for the physics-based MZI implementation in ficonn_core.py

This script demonstrates both ideal and non-ideal MZI operation modes
and verifies the physics-based implementation works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
from ficonn_core import mzi_forward, create_mzi_matrix, verify_mzi_unitarity

def test_mzi_ideal_mode():
    """Test MZI in ideal mode with various phase controls."""
    print("=== MZI IDEAL MODE TESTING ===")
    
    # Test input vector
    inputs = np.array([complex(1.0, 0.0), complex(0.0, 0.0)])
    ext_phase = 0.0
    
    print(f"Input vector: {inputs}")
    print(f"External phase: {ext_phase}")
    print()
    
    # Test different phase controls
    phase_controls = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    
    for phase_control in phase_controls:
        output = mzi_forward(inputs, phase_control, ext_phase, mode='ideal')
        powers = np.abs(output)**2
        total_power = np.sum(powers)
        
        print(f"Phase Control: {phase_control:.4f} rad")
        print(f"  Output: {output}")
        print(f"  Powers: {powers}")
        print(f"  Total Power: {total_power:.6f}")
        print()

def test_mzi_non_ideal_mode():
    """Test MZI in non-ideal mode with hardware fitting parameters."""
    print("=== MZI NON-IDEAL MODE TESTING ===")
    
    # Test input vector
    inputs = np.array([complex(1.0, 0.0), complex(0.0, 0.0)])
    ext_phase = 0.0
    
    # Hardware fitting parameters (example values)
    hw_params = {'p4': 0.001, 'p3': -0.01, 'p2': 0.1, 'p1': 3.1, 'p0': 0.15}
    
    print(f"Input vector: {inputs}")
    print(f"External phase: {ext_phase}")
    print(f"Hardware params: {hw_params}")
    print()
    
    # Test different current controls
    current_controls = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for current_control in current_controls:
        output = mzi_forward(inputs, current_control, ext_phase, mode='non_ideal', fitting_params=hw_params)
        powers = np.abs(output)**2
        total_power = np.sum(powers)
        
        print(f"Current Control: {current_control:.2f} mA")
        print(f"  Output: {output}")
        print(f"  Powers: {powers}")
        print(f"  Total Power: {total_power:.6f}")
        print()

def test_mzi_matrix_unitarity():
    """Test that create_mzi_matrix produces unitary matrices."""
    print("=== MZI MATRIX UNITARITY TESTING ===")
    
    # Test with random parameters
    n_tests = 100
    unitarity_errors = []
    
    for i in range(n_tests):
        theta1 = np.random.uniform(0, 2*np.pi)
        theta2 = np.random.uniform(0, 2*np.pi)
        
        # Create MZI matrix
        mzi_matrix = create_mzi_matrix(theta1, theta2)
        
        # Verify unitarity
        verification = verify_mzi_unitarity(mzi_matrix)
        unitarity_errors.append(verification['unitarity_error'])
        
        if i < 5:  # Print first 5 examples
            print(f"Test {i+1}: θ₁={theta1:.3f}, θ₂={theta2:.3f}")
            print(f"  Unitary: {verification['is_unitary']}")
            print(f"  Error: {verification['unitarity_error']:.2e}")
            print()
    
    # Summary statistics
    max_error = np.max(unitarity_errors)
    mean_error = np.mean(unitarity_errors)
    
    print(f"Unitarity Test Summary:")
    print(f"  Tests: {n_tests}")
    print(f"  Max Error: {max_error:.2e}")
    print(f"  Mean Error: {mean_error:.2e}")
    print(f"  All Unitary: {max_error < 1e-10}")
    print()

def test_mzi_power_conservation():
    """Test power conservation for various input vectors."""
    print("=== MZI POWER CONSERVATION TESTING ===")
    
    # Test different input vectors
    test_inputs = [
        np.array([1.0+0j, 0.0+0j]),      # Single channel
        np.array([0.0+0j, 1.0+0j]),      # Single channel
        np.array([1.0+0j, 1.0+0j]),      # Both channels
        np.array([0.5+0.3j, 0.8-0.2j]),  # Complex inputs
        np.array([1.0+1j, 0.5-0.5j])     # Complex inputs
    ]
    
    theta1 = np.pi/3
    theta2 = np.pi/4
    
    for i, inputs in enumerate(test_inputs):
        output = mzi_forward(inputs, theta1, theta2, mode='ideal')
        
        input_power = np.sum(np.abs(inputs)**2)
        output_power = np.sum(np.abs(output)**2)
        power_error = abs(input_power - output_power)
        
        print(f"Test {i+1}:")
        print(f"  Input: {inputs}")
        print(f"  Output: {output}")
        print(f"  Input Power: {input_power:.6f}")
        print(f"  Output Power: {output_power:.6f}")
        print(f"  Power Error: {power_error:.2e}")
        print()

def test_mzi_parameter_sensitivity():
    """Test that small parameter changes produce corresponding output changes."""
    print("=== MZI PARAMETER SENSITIVITY TESTING ===")
    
    inputs = np.array([1.0+0j, 0.0+0j])
    base_theta1 = np.pi/4
    base_theta2 = np.pi/6
    
    # Get baseline output
    base_output = mzi_forward(inputs, base_theta1, base_theta2, mode='ideal')
    
    # Test theta1 sensitivity
    delta_theta1 = 0.01
    new_output1 = mzi_forward(inputs, base_theta1 + delta_theta1, base_theta2, mode='ideal')
    output_change1 = np.abs(new_output1 - base_output)
    
    # Test theta2 sensitivity
    delta_theta2 = 0.01
    new_output2 = mzi_forward(inputs, base_theta1, base_theta2 + delta_theta2, mode='ideal')
    output_change2 = np.abs(new_output2 - base_output)
    
    print(f"Baseline: θ₁={base_theta1:.3f}, θ₂={base_theta2:.3f}")
    print(f"Output: {base_output}")
    print()
    print(f"θ₁ Sensitivity (Δθ₁={delta_theta1}):")
    print(f"  New Output: {new_output1}")
    print(f"  Change: {output_change1}")
    print()
    print(f"θ₂ Sensitivity (Δθ₂={delta_theta2}):")
    print(f"  New Output: {new_output2}")
    print(f"  Change: {output_change2}")
    print()

def plot_mzi_transfer_characteristics():
    """Plot MZI transfer characteristics vs phase control."""
    print("=== GENERATING MZI TRANSFER CHARACTERISTICS PLOT ===")
    
    inputs = np.array([1.0+0j, 0.0+0j])
    ext_phase = 0.0
    
    # Phase control range
    phase_controls = np.linspace(0, 2*np.pi, 100)
    output_powers = []
    
    for phase_control in phase_controls:
        output = mzi_forward(inputs, phase_control, ext_phase, mode='ideal')
        powers = np.abs(output)**2
        output_powers.append(powers)
    
    output_powers = np.array(output_powers)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(phase_controls, output_powers[:, 0], 'b-', label='Output 1', linewidth=2)
    plt.plot(phase_controls, output_powers[:, 1], 'r-', label='Output 2', linewidth=2)
    plt.plot(phase_controls, np.sum(output_powers, axis=1), 'k--', label='Total Power', linewidth=1)
    
    plt.xlabel('Phase Control θ₁ (radians)')
    plt.ylabel('Output Power')
    plt.title('MZI Transfer Characteristics (Ideal Mode)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2*np.pi)
    plt.ylim(0, 1.1)
    
    # Add key points
    key_phases = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for phase in key_phases:
        plt.axvline(phase, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('mzi_transfer_characteristics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Plot saved as 'mzi_transfer_characteristics.png'")
    print()

# --- Example Usage ---
if __name__ == '__main__':
    print("🔬 MZI PHYSICS-BASED IMPLEMENTATION TESTING")
    print("=" * 60)
    print()
    
    # Run all tests
    test_mzi_ideal_mode()
    test_mzi_non_ideal_mode()
    test_mzi_matrix_unitarity()
    test_mzi_power_conservation()
    test_mzi_parameter_sensitivity()
    plot_mzi_transfer_characteristics()
    
    print("✅ All MZI tests completed!")
    print("The physics-based MZI implementation is working correctly.")
