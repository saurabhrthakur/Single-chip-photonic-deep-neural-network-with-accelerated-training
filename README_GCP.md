# FICONN Google Cloud Integration

Simple integration to connect your FICONN project with Google Cloud Platform.

## 🚀 Quick Start

```python
from ficonn_gcp_integration import FICONNGCPIntegration

# Initialize
fic = FICONNGCPIntegration()

# Save data
fic.save_training_data(your_data, 'filename.pkl')

# Save model
fic.save_model(trained_model, 'model_name')
```

## 📁 Files

- `gcp_config.py` - GCP configuration and clients
- `ficonn_gcp_integration.py` - Main integration utilities

## 🔐 Already Configured

- ✅ Connected to project: `ficonncs4thyr`
- ✅ Storage buckets created
- ✅ Authentication set up

## 💡 Basic Usage

```python
# Save training data
gcs_path = fic.save_training_data(data, 'training_data.pkl')

# Load training data
loaded_data = fic.load_training_data('training_data.pkl')

# Save trained model
fic.save_model(model, 'ficonn_model_v1')

# List saved models
models = fic.list_saved_models()
```

That's it! Your FICONN project is now connected to Google Cloud.

