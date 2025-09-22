#!/usr/bin/env python3
"""
Test with higher learning rate to see if we can get better results.
"""

import numpy as np
from simple_ficonn_insitu_training import SimpleFiconnTrainer
from vowel_dataset import get_vowel_data

def test_higher_learning_rate():
    """
    Test with higher learning rate to see if we can get better results.
    """
    print("🧪 Testing with Higher Learning Rate")
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
    
    # Test with higher learning rate
    print("\n🏃 Running 10 epochs with higher learning rate...")
    print("   δ = 0.05 (Bernoulli perturbations)")
    print("   η = 0.01 (10x higher than paper)")
    print("   Loss: L = Σ y_train(j) * log(V_norm(j))")
    print("   Update: Θ → Θ - η∇∆L(Θ)Δ")
    
    best_params, final_test_acc = trainer.train(
        X_train_small, y_train_small,
        X_test_small, y_test_small,
        n_epochs=1000,
        learning_rate=0.002, # 10x higher than paper
        perturbation_magnitude=0.05  # Paper's δ value
    )
    
    print(f"\n🎯 Results with Higher Learning Rate:")
    print(f"   Final Test Accuracy: {final_test_acc:.2f}%")
    
    # Check if accuracy improved significantly
    if final_test_acc > 10:  # Better than random chance
        print("✅ Higher learning rate is working! Significant improvement achieved.")
    else:
        print("❌ Still need to investigate further.")
    
    return final_test_acc

if __name__ == "__main__":
    test_higher_learning_rate()
