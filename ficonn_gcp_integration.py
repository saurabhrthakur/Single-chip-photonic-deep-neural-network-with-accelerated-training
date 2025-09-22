#!/usr/bin/env python3
"""
FICONN Google Cloud Integration

Core integration utilities for connecting your FICONN project with Google Cloud Platform.
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from gcp_config import gcp_config
from google.cloud import storage, aiplatform
from google.cloud.aiplatform import Model, Endpoint

class FICONNGCPIntegration:
    """
    Google Cloud integration for FICONN project
    """
    
    def __init__(self):
        """Initialize GCP integration"""
        self.gcp = gcp_config
        self.bucket_name = f"ficonn-{self.gcp.project_id}"
        self.model_bucket_name = f"ficonn-models-{self.gcp.project_id}"
        
        # Ensure buckets exist
        self._setup_storage()
    
    def _setup_storage(self):
        """Set up Google Cloud Storage buckets"""
        try:
            # Create main bucket for data
            if self.bucket_name not in self.gcp.list_buckets():
                self.gcp.create_bucket(self.bucket_name)
                print(f"✅ Created bucket: {self.bucket_name}")
            
            # Create bucket for models
            if self.model_bucket_name not in self.gcp.list_buckets():
                self.gcp.create_bucket(self.model_bucket_name)
                print(f"✅ Created bucket: {self.model_bucket_name}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not create buckets: {e}")
    
    def save_training_data(self, data, filename):
        """
        Save training data to Google Cloud Storage
        
        Args:
            data: Data to save (numpy array, dict, etc.)
            filename: Name of the file in storage
        """
        try:
            # Save locally first
            local_path = f"temp_{filename}"
            
            if isinstance(data, np.ndarray):
                np.save(local_path, data)
            else:
                with open(local_path, 'wb') as f:
                    pickle.dump(data, f)
            
            # Upload to GCS
            blob_name = f"training_data/{filename}"
            blob = self.gcp.upload_file(self.bucket_name, local_path, blob_name)
            
            # Clean up local file
            os.remove(local_path)
            
            print(f"✅ Saved training data: gs://{self.bucket_name}/{blob_name}")
            return f"gs://{self.bucket_name}/{blob_name}"
            
        except Exception as e:
            print(f"❌ Error saving training data: {e}")
            return None
    
    def load_training_data(self, filename):
        """
        Load training data from Google Cloud Storage
        
        Args:
            filename: Name of the file in storage
            
        Returns:
            Loaded data
        """
        try:
            blob_name = f"training_data/{filename}"
            local_path = f"temp_{filename}"
            
            # Download from GCS
            self.gcp.download_file(self.bucket_name, blob_name, local_path)
            
            # Load data
            if filename.endswith('.npy'):
                data = np.load(local_path)
            else:
                with open(local_path, 'rb') as f:
                    data = pickle.load(f)
            
            # Clean up local file
            os.remove(local_path)
            
            print(f"✅ Loaded training data: gs://{self.bucket_name}/{blob_name}")
            return data
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return None
    
    def save_model(self, model, model_name, metadata=None):
        """
        Save a trained model to Google Cloud Storage
        
        Args:
            model: Trained model object
            model_name: Name for the model
            metadata: Additional metadata about the model
        """
        try:
            # Save model locally
            local_path = f"temp_{model_name}.pkl"
            with open(local_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Upload to GCS
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blob_name = f"models/{model_name}_{timestamp}.pkl"
            blob = self.gcp.upload_file(self.model_bucket_name, local_path, blob_name)
            
            # Save metadata
            if metadata:
                metadata['model_path'] = f"gs://{self.model_bucket_name}/{blob_name}"
                metadata['timestamp'] = timestamp
                metadata['model_name'] = model_name
                
                metadata_path = f"temp_{model_name}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                metadata_blob = f"models/{model_name}_{timestamp}_metadata.json"
                self.gcp.upload_file(self.model_bucket_name, metadata_path, metadata_blob)
                os.remove(metadata_path)
            
            # Clean up local file
            os.remove(local_path)
            
            print(f"✅ Saved model: gs://{self.model_bucket_name}/{blob_name}")
            return f"gs://{self.model_bucket_name}/{blob_name}"
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return None
    
    def deploy_model_to_ai_platform(self, model_path, model_name):
        """
        Deploy a model to Google Cloud AI Platform
        
        Args:
            model_path: GCS path to the model
            model_name: Name for the deployed model
        """
        try:
            # Initialize AI Platform
            aiplatform.init(project=self.gcp.project_id, location=self.gcp.region)
            
            # Create model
            model = Model(
                display_name=model_name,
                project=self.gcp.project_id,
                location=self.gcp.region
            )
            
            # Deploy model
            endpoint = model.deploy(
                machine_type="n1-standard-2",
                accelerator_type="NVIDIA_TESLA_T4",
                accelerator_count=1
            )
            
            print(f"✅ Model deployed successfully!")
            print(f"   Model: {model_name}")
            print(f"   Endpoint: {endpoint.name}")
            return endpoint
            
        except Exception as e:
            print(f"❌ Error deploying model: {e}")
            return None
    
    def list_saved_models(self):
        """List all saved models in storage"""
        try:
            bucket = self.gcp.storage_client.bucket(self.model_bucket_name)
            blobs = bucket.list_blobs(prefix="models/")
            
            models = []
            for blob in blobs:
                if blob.name.endswith('.pkl'):
                    models.append(blob.name)
            
            return models
            
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []
