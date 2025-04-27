# anomaly_detector_train.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib  # For saving the scaler
import json    # For saving scaler params for JS

print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
CSV_FILE = 'invoices.csv'
# Select features relevant for numerical/date anomaly detection
# We'll skip categorical/text fields for this autoencoder approach
FEATURES = ['product_id', 'qty', 'amount', 'invoice_date']
DATE_COLUMN = 'invoice_date'
DATE_FORMAT = '%d/%m/%Y' # Adjust if your date format is different

MODEL_SAVE_PATH = 'invoice_anomaly_model.h5'
SCALER_SAVE_PATH = 'scaler.joblib'
SCALER_PARAMS_SAVE_PATH = 'scaler_params.json' # For JS
THRESHOLD_SAVE_PATH = 'anomaly_threshold.json' # For JS

# --- 1. Data Loading and Preparation ---
print(f"Loading data from {CSV_FILE}...")
try:
    df = pd.read_csv(CSV_FILE)
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {CSV_FILE}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print("Selected features:", FEATURES)
df_selected = df[FEATURES].copy()

# Convert date column to numerical (Unix timestamp)
print(f"Converting date column '{DATE_COLUMN}'...")
try:
    # Using errors='coerce' will turn unparseable dates into NaT (Not a Time)
    df_selected[DATE_COLUMN] = pd.to_datetime(df_selected[DATE_COLUMN], format=DATE_FORMAT, errors='coerce')
    # Drop rows where date conversion failed
    original_rows = len(df_selected)
    df_selected.dropna(subset=[DATE_COLUMN], inplace=True)
    dropped_rows = original_rows - len(df_selected)
    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows due to invalid date format.")

    # Convert to Unix timestamp (seconds since epoch)
    # Divide by a large number to keep it roughly in scale with other features initially
    # Scaling will handle the final range adjustment.
    df_selected[DATE_COLUMN + '_timestamp'] = df_selected[DATE_COLUMN].astype(np.int64) // 10**9
    df_selected = df_selected.drop(columns=[DATE_COLUMN])
    print("Date column converted to timestamp.")
except KeyError:
     print(f"Error: Date column '{DATE_COLUMN}' not found in the features list or DataFrame.")
     exit()
except Exception as e:
    print(f"Error during date conversion: {e}")
    exit()

# Ensure all selected columns are numeric, coerce errors
print("Ensuring numeric types...")
for col in df_selected.columns:
     # Use errors='coerce' to turn non-numeric values into NaN
     df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

# Handle potential NaNs introduced by coercion or original data
if df_selected.isnull().values.any():
    print("Warning: NaN values found. Filling with median...")
    # Simple strategy: fill with median. More complex imputation could be used.
    df_selected.fillna(df_selected.median(), inplace=True)

print(f"Data shape after preprocessing: {df_selected.shape}")
if df_selected.empty:
    print("Error: No data left after preprocessing. Check data quality and feature selection.")
    exit()

# Scale numerical features to [0, 1]
print("Scaling data...")
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_selected)
print("Data scaled successfully.")

# Save scaler and its parameters for JS usage
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"Scaler saved to {SCALER_SAVE_PATH}")

# Extract min_ and scale_ (1 / (max - min)) for JS
# Note: scaler.min_ corresponds to data_min_
# Note: scaler.scale_ corresponds to 1 / (data_max_ - data_min_)
# The formula in JS will be: X_scaled = (X - min_) * scale_
scaler_params = {
    'min_': scaler.min_.tolist(),
    'scale_': scaler.scale_.tolist(),
    'feature_names': df_selected.columns.tolist() # Good practice to know the order
}
with open(SCALER_PARAMS_SAVE_PATH, 'w') as f:
    json.dump(scaler_params, f, indent=4)
print(f"Scaler parameters saved to {SCALER_PARAMS_SAVE_PATH}")

# Split data (optional for autoencoder thresholding, but good practice)
# We train on the entire dataset assuming it's mostly normal
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# --- 2. Model Building (Autoencoder) ---
print("Building Autoencoder model...")
input_dim = X_train.shape[1]
encoding_dim = max(2, input_dim // 2) # Choose a bottleneck size (at least 2)
latent_dim = max(1, encoding_dim // 2)

autoencoder = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(input_dim,)),
        # Encoder
        keras.layers.Dense(encoding_dim, activation="relu"),
        keras.layers.Dense(latent_dim, activation="relu", name="bottleneck"),
        # Decoder
        keras.layers.Dense(encoding_dim, activation="relu"),
        keras.layers.Dense(input_dim, activation="sigmoid"), # Sigmoid for [0, 1] scaled data
    ]
)

autoencoder.compile(optimizer="adam", loss="mae") # Mean Absolute Error is good for reconstruction
autoencoder.summary()

# --- 3. Model Training ---
print("Training model...")
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = autoencoder.fit(
    X_train,
    X_train, # Autoencoder learns to reconstruct its input
    epochs=100,
    batch_size=32,
    validation_data=(X_test, X_test), # Use test set for validation during training
    callbacks=[early_stopping],
    shuffle=True,
    verbose=1 # Set to 1 or 2 for progress, 0 for silent
)
print("Model training complete.")

# --- 4. Determine Anomaly Threshold ---
print("Determining anomaly threshold...")
# Get reconstruction errors on the *training* data
train_predictions = autoencoder.predict(X_train)
train_mae_loss = np.mean(np.abs(train_predictions - X_train), axis=1)

# Set threshold as a percentile of training errors (e.g., 99th)
# This means 1% of the training data would be flagged as anomalous
threshold = np.percentile(train_mae_loss, 99)
print(f"Anomaly threshold (99th percentile of training MAE): {threshold}")

# Save the threshold
threshold_data = {'threshold': threshold}
with open(THRESHOLD_SAVE_PATH, 'w') as f:
    json.dump(threshold_data, f)
print(f"Anomaly threshold saved to {THRESHOLD_SAVE_PATH}")


# --- 5. Save the Trained Model ---
print(f"Saving trained model to {MODEL_SAVE_PATH}...")
autoencoder.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

print("\n--- Process Complete ---")
print(f"Model saved as: {MODEL_SAVE_PATH}")
print(f"Scaler saved as: {SCALER_SAVE_PATH}")
print(f"Scaler params for JS saved as: {SCALER_PARAMS_SAVE_PATH}")
print(f"Anomaly threshold for JS saved as: {THRESHOLD_SAVE_PATH}")