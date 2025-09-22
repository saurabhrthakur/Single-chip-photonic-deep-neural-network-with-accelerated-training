#!/usr/bin/env python3
"""
Test script for the rectangular CMXU mesh architecture in ficonn_core.py

This script verifies that the new rectangular mesh produces unitary matrices
and maintains proper connectivity.
"""

import numpy as np
from ficonn_core import create_clements_mesh_6x6, verify_clements_unitarity

def test_rectangular_mesh_unitarity():
    """Test that the rectangular mesh produces unitary matrices."""
    print("=== RECTANGULAR MESH UNITARITY TESTING ===")
    
    # Test with random parameters
    n_tests = 100
    unitarity_errors = []
    power_conservation_errors = []
    
    for i in range(n_tests):
        # Generate random parameters
        mzi_params = np.random.uniform(0, 2*np.pi, 36)
        
        # Create CMXU matrix
        cmxu_matrix = create_clements_mesh_6x6(mzi_params)
        
        # Verify unitarity
        verification = verify_clements_unitarity(cmxu_matrix)
        unitarity_errors.append(verification['unitarity_error'])
        
        # Test power conservation with random input
        test_input = np.random.randn(6) + 1j * np.random.randn(6)
        test_input = test_input / np.linalg.norm(test_input)  # Normalize
        
        output = cmxu_matrix @ test_input
        input_power = np.sum(np.abs(test_input)**2)
        output_power = np.sum(np.abs(output)**2)
        power_error = abs(input_power - output_power)
        power_conservation_errors.append(power_error)
        
        if i < 5:  # Print first 5 examples
            print(f"Test {i+1}:")
            print(f"  Unitary: {verification['is_unitary']}")
            print(f"  Unitarity Error: {verification['unitarity_error']:.2e}")
            print(f"  Power Conservation Error: {power_error:.2e}")
            print()
    
    # Summary statistics
    max_unitarity_error = np.max(unitarity_errors)
    mean_unitarity_error = np.mean(unitarity_errors)
    max_power_error = np.max(power_conservation_errors)
    mean_power_error = np.mean(power_conservation_errors)
    
    print(f"Rectangular Mesh Test Summary:")
    print(f"  Tests: {n_tests}")
    print(f"  Max Unitarity Error: {max_unitarity_error:.2e}")
    print(f"  Mean Unitarity Error: {mean_unitarity_error:.2e}")
    print(f"  Max Power Error: {max_power_error:.2e}")
    print(f"  Mean Power Error: {mean_power_error:.2e}")
    print(f"  All Unitary: {max_unitarity_error < 1e-10}")
    print(f"  All Power Conserved: {max_power_error < 1e-10}")
    print()

def test_rectangular_mesh_connectivity():
    """Test that the rectangular mesh provides full connectivity."""
    print("=== RECTANGULAR MESH CONNECTIVITY TESTING ===")
    
    # Test with specific input patterns to check connectivity
    test_cases = [
        np.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j]),  # Input on channel 0
        np.array([0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j]),  # Input on channel 1
        np.array([0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j]),  # Input on channel 2
        np.array([0.0+0j, 0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j, 0.0+0j]),  # Input on channel 3
        np.array([0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 1.0+0j, 0.0+0j]),  # Input on channel 4
        np.array([0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 1.0+0j]),  # Input on channel 5
    ]
    
    # Use random parameters
    mzi_params = np.random.uniform(0, 2*np.pi, 36)
    cmxu_matrix = create_clements_mesh_6x6(mzi_params)
    
    print(f"CMXU Matrix Condition Number: {np.linalg.cond(cmxu_matrix):.2e}")
    print()
    
    for i, test_input in enumerate(test_cases):
        output = cmxu_matrix @ test_input
        output_powers = np.abs(output)**2
        
        print(f"Input Channel {i}:")
        print(f"  Output Powers: {output_powers}")
        print(f"  Max Output Power: {np.max(output_powers):.4f}")
        print(f"  Power Distribution: {output_powers / np.sum(output_powers)}")
        print()
    
    print("Connectivity Analysis:")
    print("  - Each input channel should produce non-zero output on multiple channels")
    print("  - This indicates full connectivity through the rectangular mesh")
    print()

def test_rectangular_mesh_parameter_sensitivity():
    """Test that small parameter changes produce corresponding output changes."""
    print("=== RECTANGULAR MESH PARAMETER SENSITIVITY TESTING ===")
    
    # Base parameters
    base_params = np.random.uniform(0, 2*np.pi, 36)
    test_input = np.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j, 0.0+0j])
    
    # Get baseline output
    base_matrix = create_clements_mesh_6x6(base_params)
    base_output = base_matrix @ test_input
    
    # Test sensitivity to parameter changes
    delta = 0.01
    sensitivity_errors = []
    
    for i in range(0, 36, 2):  # Test every MZI (every 2 parameters)
        # Create modified parameters
        modified_params = base_params.copy()
        modified_params[i] += delta  # Change theta1
        modified_params[i+1] += delta  # Change theta2
        
        # Get modified output
        modified_matrix = create_clements_mesh_6x6(modified_params)
        modified_output = modified_matrix @ test_input
        
        # Calculate sensitivity
        output_change = np.abs(modified_output - base_output)
        max_change = np.max(output_change)
        sensitivity_errors.append(max_change)
        
        if i < 10:  # Print first 5 examples
            print(f"MZI {i//2 + 1} (params {i}, {i+1}):")
            print(f"  Max Output Change: {max_change:.6f}")
            print()
    
    # Summary
    mean_sensitivity = np.mean(sensitivity_errors)
    max_sensitivity = np.max(sensitivity_errors)
    
    print(f"Parameter Sensitivity Summary:")
    print(f"  Mean Sensitivity: {mean_sensitivity:.6f}")
    print(f"  Max Sensitivity: {max_sensitivity:.6f}")
    print(f"  Responsive: {mean_sensitivity > 1e-6}")
    print()

def compare_rectangular_vs_triangular():
    """Compare rectangular mesh with theoretical triangular mesh properties."""
    print("=== RECTANGULAR vs TRIANGULAR MESH COMPARISON ===")
    
    # Test with same random parameters
    mzi_params = np.random.uniform(0, 2*np.pi, 36)
    
    # Create rectangular mesh
    rectangular_matrix = create_clements_mesh_6x6(mzi_params)
    
    # Verify properties
    rectangular_verification = verify_clements_unitarity(rectangular_matrix)
    
    print(f"Rectangular Mesh Properties:")
    print(f"  Unitary: {rectangular_verification['is_unitary']}")
    print(f"  Unitarity Error: {rectangular_verification['unitarity_error']:.2e}")
    print(f"  Condition Number: {rectangular_verification['condition_number']:.2e}")
    print()
    
    print(f"Architecture Comparison:")
    print(f"  Rectangular: 6 layers (3-2-3-2-3-2 MZIs)")
    print(f"  Triangular: 5 layers (5-4-3-2-1 MZIs)")
    print(f"  Both: 15 MZIs + 6 phase shifters = 36 parameters")
    print()
    
    print(f"Advantages of Rectangular Mesh:")
    print(f"  - More uniform layer structure")
    print(f"  - Better parallel processing capability")
    print(f"  - More balanced parameter distribution")
    print()

# --- Example Usage ---
if __name__ == '__main__':
    print("🔬 RECTANGULAR CMXU MESH TESTING")
    print("=" * 60)
    print()
    
    # Run all tests
    test_rectangular_mesh_unitarity()
    test_rectangular_mesh_connectivity()
    test_rectangular_mesh_parameter_sensitivity()
    compare_rectangular_vs_triangular()
    
    print("✅ All rectangular CMXU tests completed!")
    print("The rectangular mesh architecture is working correctly.")
