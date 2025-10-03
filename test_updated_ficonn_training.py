#!/usr/bin/env python3
"""
Test the updated ficonn_in_situ_training.py with paper's exact protocol
"""

import numpy as np
import time
from ficonn_in_situ_training import FICONNInSituTrainer
from vowel_dataset import get_vowel_data

def test_updated_ficonn_training():
    """
    Test the updated FiCONN in-situ training with paper's exact protocol
    """
    print("🧪 Testing Updated FiCONN In-Situ Training")
    print("=" * 60)
    
    # Load dataset
    X_train, X_test, y_train, y_test, _, class_names = get_vowel_data()
    print(f"Dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Classes: {class_names}")
    
    # Initialize trainer
    trainer = FICONNInSituTrainer(n_channels=6, n_layers=3)
    
    # Run training with paper's exact parameters
    print("\n🚀 Starting Training with Paper's Protocol:")
    print("   - Learning rate: 0.002 (as per paper)")
    print("   - Perturbation magnitude: 0.05 (as per paper)")
    print("   - Bernoulli ±δ perturbations")
    print("   - Loss formula: L = Σ y_train(j) * log(V_norm(j))")
    print("   - Update rule: Θ → Θ - η∇∆L(Θ)Δ")
    
    start_time = time.time()
    
    try:
        best_params, final_test_acc = trainer.train(
            X_train, y_train,
            X_test, y_test,
            n_epochs=100,  # Start with fewer epochs for testing
            learning_rate=0.002,  # Paper's exact learning rate
            perturbation_magnitude=0.05  # Paper's exact perturbation magnitude
        )
        
        training_time = time.time() - start_time
        
        print(f"\n⏱️ Training completed in {training_time:.2f} seconds")
        print(f"🎯 Final Test Accuracy: {final_test_acc:.2f}%")
        
        # Plot training history
        trainer.plot_training_history('updated_ficonn_training.png')
        
        return best_params, final_test_acc
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def compare_with_simple_version():
    """
    Compare updated version with simple version
    """
    print("\n🔬 Comparing with Simple Version:")
    print("-" * 40)
    
    # Load dataset
    X_train, X_test, y_train, y_test, _, class_names = get_vowel_data()
    
    # Use smaller subset for quick comparison
    X_train_small = X_train[:100]
    y_train_small = y_train[:100]
    X_test_small = X_test[:50]
    y_test_small = y_test[:50]
    
    print(f"Using subset: {len(X_train_small)} train, {len(X_test_small)} test samples")
    
    # Test updated version
    print("\n📊 Testing Updated Version (10 epochs):")
    trainer_updated = FICONNInSituTrainer(n_channels=6, n_layers=3)
    
    start_time = time.time()
    try:
        _, acc_updated = trainer_updated.train(
            X_train_small, y_train_small,
            X_test_small, y_test_small,
            n_epochs=10,
            learning_rate=0.002,
            perturbation_magnitude=0.05
        )
        time_updated = time.time() - start_time
        print(f"✅ Updated: {acc_updated:.2f}% accuracy in {time_updated:.2f}s")
    except Exception as e:
        print(f"❌ Updated failed: {e}")
        acc_updated = 0
        time_updated = float('inf')
    
    # Test simple version
    print("\n📊 Testing Simple Version (10 epochs):")
    from simple_ficonn_insitu_training import SimpleFiconnTrainer
    trainer_simple = SimpleFiconnTrainer(n_channels=6, n_layers=3)
    
    start_time = time.time()
    try:
        _, acc_simple = trainer_simple.train(
            X_train_small, y_train_small,
            X_test_small, y_test_small,
            n_epochs=10,
            learning_rate=0.002,
            perturbation_magnitude=0.05
        )
        time_simple = time.time() - start_time
        print(f"✅ Simple: {acc_simple:.2f}% accuracy in {time_simple:.2f}s")
    except Exception as e:
        print(f"❌ Simple failed: {e}")
        acc_simple = 0
        time_simple = float('inf')
    
    # Results
    print("\n📈 Comparison Results:")
    print("=" * 40)
    print(f"Updated Version:  {acc_updated:.2f}% accuracy, {time_updated:.2f}s")
    print(f"Simple Version:   {acc_simple:.2f}% accuracy, {time_simple:.2f}s")
    
    if time_updated != float('inf') and time_simple != float('inf'):
        speedup = time_simple / time_updated
        print(f"Speed Ratio: {speedup:.2f}x")
        
        if speedup > 1.1:
            print("🚀 Updated version is faster!")
        elif speedup < 0.9:
            print("⚠️ Updated version is slower")
        else:
            print("✅ Similar performance")

if __name__ == "__main__":
    # Test updated training
    best_params, final_acc = test_updated_ficonn_training()
    
    # Compare with simple version
    compare_with_simple_version()
    
    print(f"\n🎯 Final Result: {final_acc:.2f}% test accuracy")
