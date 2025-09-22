#!/usr/bin/env python3
"""
Google Cloud Platform configuration for FICONN project
"""

import os
from google.cloud import storage, compute_v1, aiplatform
from google.auth import default

class GCPConfig:
    """Google Cloud Platform configuration and client management"""
    
    def __init__(self):
        self.credentials, self.project_id = default()
        self.region = "us-central1"  # Default region, can be changed
        self.zone = "us-central1-a"  # Default zone, can be changed
        
        # Initialize clients
        self.storage_client = storage.Client()
        self.compute_client = compute_v1.InstancesClient()
        
        # Set AI Platform project and location
        aiplatform.init(project=self.project_id, location=self.region)
    
    def get_project_info(self):
        """Get current project information"""
        return {
            "project_id": self.project_id,
            "region": self.region,
            "zone": self.zone,
            "account": self.credentials.service_account_email if hasattr(self.credentials, 'service_account_email') else 'User account'
        }
    
    def list_buckets(self, max_results=10):
        """List storage buckets"""
        buckets = list(self.storage_client.list_buckets(max_results=max_results))
        return [bucket.name for bucket in buckets]
    
    def create_bucket(self, bucket_name, location=None):
        """Create a new storage bucket"""
        if location is None:
            location = self.region
        
        bucket = self.storage_client.bucket(bucket_name)
        bucket.location = location
        bucket.create()
        return bucket
    
    def upload_file(self, bucket_name, source_file, destination_blob_name):
        """Upload a file to Google Cloud Storage"""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file)
        return blob
    
    def download_file(self, bucket_name, source_blob_name, destination_file):
        """Download a file from Google Cloud Storage"""
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file)
        return destination_file

# Global instance
gcp_config = GCPConfig()
