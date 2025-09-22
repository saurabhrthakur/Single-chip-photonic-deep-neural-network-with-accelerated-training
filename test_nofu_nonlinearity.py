#!/usr/bin/env python3
"""
Test NOFU Nonlinearity Response

This script tests the NOFU activation function to verify it shows proper nonlinear behavior
as expected from the paper's Figure 3d.
"""

import numpy as np
import matplotlib.pyplot as plt
from ficonn_core import nofu_activation_with_bias

def test_nofu_response():
    """
    Test NOFU activation function with different parameter sets to verify nonlinearity.
    """
    print("🧪 Testing NOFU Nonlinearity Response")
    print("=" * 50)
    
    # Test parameters from paper's Figure 3d
    test_params = [
        {"beta": 0.2, "delta_lambda": 0.1, "V_B": 0.8, "name": "β=0.2, Δλ=0.1nm"},
        {"beta": 0.4, "delta_lambda": 0.1, "V_B": 0.8, "name": "β=0.4, Δλ=0.1nm"},
        {"beta": 0.4, "delta_lambda": 0.2, "V_B": 0.8, "name": "β=0.4, Δλ=0.2nm"},
        {"beta": 0.2, "delta_lambda": 0.2, "V_B": 0.8, "name": "β=0.2, Δλ=0.2nm"},
        {"beta": 0.2, "delta_lambda": 0.15, "V_B": 0.8, "name": "β=0.2, Δλ=0.15nm"},
        {"beta": 0.4, "delta_lambda": 0.25, "V_B": 0.8, "name": "β=0.4, Δλ=0.25nm"},
    ]
    
    # Input power range
    input_powers = np.logspace(-6, -2, 100)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, params in enumerate(test_params):
        print(f"\nTesting {params['name']}...")
        
        # Calculate output powers
        output_powers = []
        for P_in in input_powers:
            # Create complex input field
            b_in = np.sqrt(P_in) * np.exp(1j * 0.1)  # Small phase for realism
            
            # Apply NOFU activation
            b_out = nofu_activation_with_bias(
                b_in, 
                params["beta"], 
                params["delta_lambda"], 
                params["V_B"]
            )
            
            # Calculate output power
            P_out = np.abs(b_out)**2
            output_powers.append(P_out)
        
        output_powers = np.array(output_powers)
        
        # Plot
        axes[i].plot(input_powers, output_powers, 'b-', linewidth=2, label=params['name'])
        axes[i].plot(input_powers, input_powers, 'r--', alpha=0.5, label='Linear (y=x)')
        axes[i].set_xlabel('Input Power (a.u.)')
        axes[i].set_ylabel('Output Power (a.u.)')
        axes[i].set_title(params['name'])
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Calculate nonlinearity metrics
        linear_output = input_powers
        nonlinearity = np.mean(np.abs(output_powers - linear_output) / linear_output) * 100
        
        print(f"  Nonlinearity: {nonlinearity:.2f}%")
        print(f"  Min output: {np.min(output_powers):.4f}")
        print(f"  Max output: {np.max(output_powers):.4f}")
        print(f"  Output range: {np.max(output_powers) - np.min(output_powers):.4f}")
        
        # Check for non-monotonic behavior (important for complex nonlinearity)
        diff = np.diff(output_powers)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        print(f"  Sign changes in derivative: {sign_changes}")
        
        if sign_changes > 0:
            print(f"  ✅ Non-monotonic behavior detected (good for nonlinearity)")
        else:
            print(f"  ⚠️  Monotonic behavior (may be too linear)")
    
    plt.tight_layout()
    plt.savefig('nofu_nonlinearity_test.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ NOFU nonlinearity test completed!")
    print(f"📊 Results saved to: nofu_nonlinearity_test.png")

def test_parameter_sensitivity():
    """
    Test how sensitive NOFU is to parameter changes.
    """
    print(f"\n🔍 Testing Parameter Sensitivity")
    print("=" * 50)
    
    # Base parameters
    base_beta = 0.3
    base_delta_lambda = 0.15
    V_B = 0.8
    
    # Test input
    P_in = 1.0
    b_in = np.sqrt(P_in) * np.exp(1j * 0.1)
    
    # Test beta sensitivity
    print("\nBeta sensitivity:")
    for beta in [0.1, 0.2, 0.3, 0.4, 0.5]:
        b_out = nofu_activation_with_bias(b_in, beta, base_delta_lambda, V_B)
        P_out = np.abs(b_out)**2
        print(f"  β={beta:.1f}: P_out={P_out:.4f}")
    
    # Test delta_lambda sensitivity
    print("\nDelta lambda sensitivity:")
    for delta_lambda in [0.05, 0.1, 0.15, 0.2, 0.25]:
        b_out = nofu_activation_with_bias(b_in, base_beta, delta_lambda, V_B)
        P_out = np.abs(b_out)**2
        print(f"  Δλ={delta_lambda:.2f}: P_out={P_out:.4f}")

if __name__ == "__main__":
    test_nofu_response()
    test_parameter_sensitivity()
