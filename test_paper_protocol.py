#!/usr/bin/env python3
"""
Quick test of the paper's exact training protocol.
"""

import numpy as np
from simple_ficonn_insitu_training import SimpleFiconnTrainer
from vowel_dataset import get_vowel_data

def test_paper_protocol():
    """
    Test the paper's exact training protocol with small dataset.
    """
    print("🧪 Testing Paper's Exact Training Protocol")
    print("=" * 50)
    
    # Load small dataset
    X_train, X_test, y_train, y_test, _, class_names = get_vowel_data()
    
    # Use very small subset for quick test
    X_train_small = X_train[:50]
    y_train_small = y_train[:50]
    X_test_small = X_test[:25]
    y_test_small = y_test[:25]
    
    print(f"Training samples: {len(X_train_small)}")
    print(f"Test samples: {len(X_test_small)}")
    
    # Initialize trainer
    trainer = SimpleFiconnTrainer(n_channels=6, n_layers=3)
    
    # Test with paper's exact parameters
    print("\n🏃 Running 10 epochs with paper's exact protocol...")
    print("   δ = 0.05 (Bernoulli perturbations)")
    print("   η = 0.002 (paper's learning rate)")
    print("   Loss: L = Σ y_train(j) * log(V_norm(j))")
    print("   Update: Θ → Θ - η∇∆L(Θ)Δ")
    
    best_params, final_test_acc = trainer.train(
        X_train_small, y_train_small,
        X_test_small, y_test_small,
        n_epochs=10,
        learning_rate=0.002,  # Paper's exact learning rate
        perturbation_magnitude=0.05  # Paper's δ value
    )
    
    print(f"\n🎯 Results with Paper's Protocol:")
    print(f"   Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Check if accuracy improved significantly
    if final_test_acc > 30:  # Better than random chance
        print("✅ Paper's protocol is working! Significant improvement achieved.")
    else:
        print("❌ Still need to investigate further.")
    
    return final_test_acc

if __name__ == "__main__":
    test_paper_protocol()
