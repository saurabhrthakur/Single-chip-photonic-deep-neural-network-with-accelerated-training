import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# ==============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ==============================================================================

def get_vowel_data(print_info=True):
    """
    Loads, cleans, and preprocesses the Hillenbrand vowel dataset using a robust
    pandas-based pipeline that correctly handles the file's true 16-column structure.
    """
    # Define file path and the CORRECT 16 column names from the file's header
    file_path = 'vowdata.dat'
    all_column_names = [
        'filename', 'duration_ms', 'f0_ss', 'F1_ss', 'F2_ss', 'F3_ss', 'F4_ss',
        'F1_20', 'F2_20', 'F3_20', 'F1_50', 'F2_50', 'F3_50', 'F1_80', 'F2_80', 'F3_80'
    ]
    
    # --- Data Loading ---
    # We load the data starting from the actual data rows (line 31), skipping the header
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    df = pd.read_csv(file_path, header=None, names=all_column_names, sep='\\s+', 
                     engine='python', skiprows=30)

    # --- Feature Extraction and Label Creation ---
    # Extract the vowel label from the filename (e.g., 'm01ae' -> 'ae')
    df['vowel'] = df['filename'].str[3:5]
    
    # The paper uses these specific 6 features for classification
    feature_cols = ['F1_ss', 'F2_ss', 'F3_ss', 'F1_50', 'F2_50', 'F3_50']
    
    # --- Data Cleaning ---
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=feature_cols + ['vowel'], inplace=True)
    
    # --- Feature and Label Finalization ---
    labels = df['vowel'].values
    features = df[feature_cols].values
    
    # --- Label Encoding ---
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    class_names = label_encoder.classes_
    
    # --- Data Splitting and Normalization ---
    X = features.astype(np.float64)

    # Shuffle the clean data consistently before splitting
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Use the exact train/test split size mentioned in the paper
    train_size = 540
    test_size = 294

    if len(X) < train_size + test_size:
        raise ValueError(f"Not enough clean data for the specified split. Need {train_size + test_size}, but have {len(X)}.")

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size : train_size + test_size]
    y_test = y[train_size : train_size + test_size]

    # Apply per-sample max-scaling normalization
    X_train_max = np.max(X_train, axis=1, keepdims=True)
    X_test_max = np.max(X_test, axis=1, keepdims=True)
    
    # Avoid division by zero
    X_train_max[X_train_max == 0] = 1
    X_test_max[X_test_max == 0] = 1
    
    X_train_normalized = X_train / X_train_max
    X_test_normalized = X_test / X_test_max

    if print_info:
        print("Dataset loaded and cleaned successfully!")
        print(f"Training samples: {len(X_train_normalized)}")
        print(f"Test samples: {len(X_test_normalized)}")
        print(f"Number of classes: {len(class_names)}\n")

    return X_train_normalized, X_test_normalized, y_train, y_test, label_encoder, class_names
