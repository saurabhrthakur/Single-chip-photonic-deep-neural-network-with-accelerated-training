#!/usr/bin/env python3
"""
Google Cloud Platform Training Script for FiCONN
Runs the in-situ training on GCP with cloud storage integration
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from ficonn_gcp_integration import FICONNGCPIntegration
from fake_insitu_training import FakeInSituTrainer
from vowel_dataset import get_vowel_data

def run_gcp_training():
    """
    Run FiCONN training on Google Cloud Platform
    """
    print("🚀 Starting FiCONN Training on Google Cloud Platform")
    print("=" * 60)
    
    # Initialize GCP integration
    try:
        gcp_integration = FICONNGCPIntegration()
        print("✅ GCP integration initialized successfully")
    except Exception as e:
        print(f"❌ GCP integration failed: {e}")
        print("Falling back to local training...")
        return run_local_training()
    
    # Load dataset
    print("\n📊 Loading Dataset...")
    X_train, X_test, y_train, y_test, _, class_names = get_vowel_data()
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Classes: {class_names}")
    
    # Save dataset to GCS for backup
    print("\n💾 Saving dataset to Google Cloud Storage...")
    dataset_metadata = {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'n_classes': len(class_names),
        'class_names': class_names.tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    gcp_integration.save_training_data({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'metadata': dataset_metadata
    }, 'vowel_dataset.pkl')
    
    # Initialize trainer
    print("\n🔧 Initializing Fake In-Situ Trainer...")
    trainer = FakeInSituTrainer(n_channels=6, n_layers=3)
    
    # Training configuration
    training_config = {
        'n_epochs': 1000,
        'learning_rate': 0.01,  # Higher learning rate for better convergence
        'algorithm': 'fake_insitu',  # Using fake in-situ training
        'dataset_size': len(X_train),
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"   Training epochs: {training_config['n_epochs']}")
    print(f"   Learning rate: {training_config['learning_rate']}")
    print(f"   Algorithm: {training_config['algorithm']}")
    
    # Start training
    print("\n🏃 Starting Training...")
    start_time = time.time()
    
    try:
        best_params, final_test_acc = trainer.train(
            X_train, y_train,
            X_test, y_test,
            n_epochs=training_config['n_epochs'],
            learning_rate=training_config['learning_rate']
        )
        
        training_time = time.time() - start_time
        
        # Training results
        results = {
            'final_test_accuracy': final_test_acc,
            'training_time_seconds': training_time,
            'training_config': training_config,
            'best_params': best_params,
            'training_history': trainer.history,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n🎯 Training Completed!")
        print(f"   Final Test Accuracy: {final_test_acc:.2f}%")
        print(f"   Training Time: {training_time:.2f} seconds")
        print(f"   Best Test Accuracy: {trainer.history['best_test_acc']:.2f}%")
        
        # Save results to GCS
        print("\n💾 Saving results to Google Cloud Storage...")
        gcp_integration.save_training_data(results, 'training_results.pkl')
        
        # Save model
        print("\n💾 Saving trained model...")
        model_metadata = {
            'final_test_accuracy': final_test_acc,
            'training_time_seconds': training_time,
            'n_epochs': training_config['n_epochs'],
            'learning_rate': training_config['learning_rate'],
            'algorithm': training_config['algorithm'],
            'dataset_size': len(X_train),
            'timestamp': datetime.now().isoformat()
        }
        
        gcp_integration.save_model({
            'best_params': best_params,
            'trainer': trainer,
            'results': results
        }, 'ficonn_trained_model', model_metadata)
        
        # List saved models
        print("\n📋 Saved Models:")
        models = gcp_integration.list_saved_models()
        for model in models:
            print(f"   - {model}")
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None

def run_local_training():
    """
    Fallback local training if GCP fails
    """
    print("\n🏠 Running Local Training (GCP fallback)...")
    
    # Load dataset
    X_train, X_test, y_train, y_test, _, class_names = get_vowel_data()
    
    # Use smaller dataset for local testing
    X_train_small = X_train[:100]
    y_train_small = y_train[:100]
    X_test_small = X_test[:50]
    y_test_small = y_test[:50]
    
    # Initialize trainer
    trainer = FakeInSituTrainer(n_channels=6, n_layers=3)
    
    # Train
    best_params, final_test_acc = trainer.train(
        X_train_small, y_train_small,
        X_test_small, y_test_small,
        n_epochs=100,  # Fewer epochs for local testing
        learning_rate=0.01
    )
    
    print(f"Local training completed. Test accuracy: {final_test_acc:.2f}%")
    return {'final_test_accuracy': final_test_acc, 'local_training': True}

def main():
    """
    Main function
    """
    print("🌐 FiCONN Google Cloud Platform Training")
    print("=" * 50)
    
    # Check if running on GCP
    if os.getenv('GOOGLE_CLOUD_PROJECT'):
        print("✅ Running on Google Cloud Platform")
        results = run_gcp_training()
    else:
        print("🏠 Running locally (not on GCP)")
        results = run_local_training()
    
    if results:
        print(f"\n✅ Training completed successfully!")
        print(f"   Final accuracy: {results.get('final_test_accuracy', 'N/A'):.2f}%")
    else:
        print(f"\n❌ Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
