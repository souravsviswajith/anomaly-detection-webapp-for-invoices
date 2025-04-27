# anomaly-detection-webapp-for-invoices

Goal: Build an anomaly detection system for invoice data using TensorFlow and deploy it for web-based checking with TensorFlow.js. The detection should be sensitive even to anomalies in single fields.

Approach:

Data Preparation (Python):

Load the invoices.csv data.

Select relevant features for anomaly detection (numerical and date fields seem most appropriate for this type of model).

Convert date/time to a numerical representation (e.g., Unix timestamp).

Scale numerical features (crucial for neural networks like Autoencoders). We'll use MinMaxScaler.

Save the scaler parameters (min and scale values) as we'll need them for preprocessing in JavaScript.

Model Building (TensorFlow/Keras - Python):

Use an Autoencoder. This unsupervised neural network learns to reconstruct "normal" input data. Anomalies are expected to have a higher reconstruction error.

Train the autoencoder on the prepared (scaled) "normal" data (we'll assume the provided CSV is mostly normal).

Calculate the reconstruction error (e.g., Mean Absolute Error - MAE) for each training sample.

Determine an anomaly threshold based on the distribution of reconstruction errors (e.g., the 99th percentile).

Save the trained model (.h5 format).

Model Conversion (Command Line):

Use tensorflowjs_converter to convert the saved Keras model (.h5) into a format usable by TensorFlow.js (model.json + weight files).

Web Interface (HTML/JavaScript with TensorFlow.js):

Create an HTML form to input invoice details.

Write JavaScript code to:

Load the converted TensorFlow.js model.

Get input values from the form.

Crucially: Preprocess the input exactly like the Python training data (convert date, scale using the saved scaler parameters).

Make a prediction (reconstruct the input) using the loaded model.

Calculate the reconstruction error (MAE) between the input and the reconstruction.

Compare the error to the pre-determined threshold.

Display whether the input is considered an anomaly or normal.



Explanation:

Imports: Necessary libraries.

Configuration: File paths and feature selection.

Data Loading: Reads the CSV.

Feature Selection: Keeps only the specified columns.

Date Conversion: Converts invoice_date to datetime objects, handles errors, calculates Unix timestamps, and drops the original date column. Timestamps are divided initially to prevent them from vastly dominating other features before scaling.

Numeric Conversion & NaN Handling: Ensures columns are numeric and fills any missing values (potentially introduced by date or numeric conversion errors) with the median.

Scaling: Uses MinMaxScaler to scale features between 0 and 1. This scaler object is saved using joblib.

Saving Scaler Parameters: Extracts the min_ and scale_ attributes from the fitted scaler and saves them to a JSON file. This is essential for correctly preprocessing data in JavaScript later.

Data Split: Splits data for model validation during training (though the threshold is based on training data reconstruction error).

Autoencoder Model: Defines a simple sequential autoencoder with Dense layers. The bottleneck layer (latent_dim) forces the network to learn a compressed representation. The output layer uses a 'sigmoid' activation because the data is scaled to [0, 1]. The loss function is 'mae'.

Training: Trains the model to reconstruct the input (X_train used as both input and target). Early stopping prevents overfitting.

Threshold Calculation: Predicts (reconstructs) the training data, calculates the MAE for each sample, and determines the 99th percentile of these errors as the anomaly threshold. This threshold is saved to a JSON file.

Model Saving: Saves the trained Keras model to an .h5 file.

Step 3: Convert Model for TensorFlow.js

Install TensorFlow.js Converter:

pip install tensorflowjs
Use code with caution.
Bash
Run the Converter: Open your terminal/command prompt, navigate to the directory where you saved the Python script and ran it, and execute:

tensorflowjs_converter --input_format keras invoice_anomaly_model.h5 tfjs_model
Use code with caution.
Bash
Replace invoice_anomaly_model.h5 if you used a different name.

tfjs_model is the directory where the converted model (model.json and .bin files) will be saved.

Step 4: Web Interface (HTML & JavaScript)

Create the following two files in a new directory (e.g., web_anomaly_checker). Copy the tfjs_model directory (created in Step 3), scaler_params.json, and anomaly_threshold.json into this same web_anomaly_checker directory.
