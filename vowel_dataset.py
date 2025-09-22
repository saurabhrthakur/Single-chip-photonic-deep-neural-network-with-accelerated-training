import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    
    # --- Filter to the 6-class subset in the specified order ---
    # Desired class order (provided by user): ae, aw, uw, er, iy, ih
    class_order = ['ae', 'aw', 'uw', 'er', 'iy', 'ih']
    df = df[df['vowel'].isin(class_order)].copy()
    
    # --- Feature and Label Finalization ---
    labels = df['vowel'].values
    features = df[feature_cols].values
    
    # --- Manual label mapping to preserve the specified order ---
    class_to_idx = {cls: idx for idx, cls in enumerate(class_order)}
    y = np.array([class_to_idx[v] for v in labels], dtype=np.int64)
    class_names = np.array(class_order)
    
    # --- Data Splitting and Normalization ---
    X = features.astype(np.float64)

    # Stratified split to maintain class balance
    if len(X) != 834:
        # This check ensures our assumption about the filtered data size is correct.
        print(f"Warning: Expected 834 samples after filtering, but found {len(X)}. The split might not be exactly 540/294.")

    # Explicitly define the train/test split sizes
    train_size = 540
    test_size = 294 # Total (834) - Train (540) = 294

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=42, stratify=y
    )

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
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes (ordered): {', '.join(class_names)}\n")
    
    # Minimal label encoder substitute with classes_ attribute
    class SimpleLabelEncoder:
        def __init__(self, classes):
            self.classes_ = np.array(classes)
    
    label_encoder = SimpleLabelEncoder(class_names)
    return X_train_normalized, X_test_normalized, y_train, y_test, label_encoder, class_names