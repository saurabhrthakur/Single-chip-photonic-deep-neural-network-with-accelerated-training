#!/usr/bin/env python3
"""
Test script to verify the new NOFU integration in ficonn_core.py
"""

import numpy as np
from ficonn_core import nofu_activation_with_bias, _apply_activation, verify_nofu_physics

def test_single_channel():
    print("=== Testing Single Channel NOFU ===")
    input_field = 1.0 + 0j
    beta = 0.2
    delta_lambda_initial = 0.1
    V_B = 0.8

    output = nofu_activation_with_bias(input_field, beta, delta_lambda_initial, V_B)
    print(f"Input: {input_field}")
    print(f"Output: {output}")
    print(f"Transmission: {np.abs(output)**2 / np.abs(input_field)**2:.4f}")
    print()

def test_vector_activation():
    print("=== Testing Vector NOFU Activation ===")
    vector = np.array([1.0 + 0j, 0.5 + 0j])
    nofu_params = np.array([0.2, 0.1, 0.1, 0.2])  # 2 betas + 2 delta_lambdas

    output_vector = _apply_activation(vector, nofu_params)
    print(f"Input vector: {vector}")
    print(f"Output vector: {output_vector}")
    print()

def test_nonlinearity():
    print("=== Testing NOFU Nonlinearity ===")
    results = verify_nofu_physics(beta=0.2, initial_detuning=0.1)
    print(f"Nonlinear: {results['is_nonlinear']}")
    print(f"Power conservation: {results['max_conservation_error']:.4f}")
    print(f"Saturation: {results['has_saturation']}")
    print()

if __name__ == "__main__":
    print("Testing New NOFU Integration\n")
    test_single_channel()
    test_vector_activation()
    test_nonlinearity()
    print("All tests completed!")
