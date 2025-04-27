# Invoice Anomaly Detection Web App (TensorFlow & TensorFlow.js)

This project demonstrates building an anomaly detection system for invoice data using a TensorFlow Autoencoder model and deploying it for real-time checking within a web browser using TensorFlow.js. It aims to detect anomalies even when only a single input field deviates significantly from the norm.

## Goal

The primary goal is to identify unusual or potentially fraudulent invoice entries by comparing them against patterns learned from a dataset of historical invoices. The anomaly detection happens directly in the user's browser for instant feedback.

## Features

*   Loads invoice data from a CSV file (`invoices.csv`).
*   Preprocesses numerical and date features suitable for model training.
*   Trains a TensorFlow/Keras Autoencoder model to learn the patterns of "normal" invoices.
*   Determines an anomaly threshold based on the model's reconstruction error on normal data.
*   Saves the trained model and necessary preprocessing parameters (scaler, threshold).
*   Converts the Keras model to TensorFlow.js format for web deployment.
*   Provides a simple web interface (`index.html`) to input new invoice details.
*   Performs real-time anomaly detection in the browser using the loaded TF.js model (`detector.js`).
*   Displays whether the entered invoice data is considered "Normal" or an "Anomaly".

## Technology Stack

*   **Backend / Model Training:**
    *   Python 3.x
    *   TensorFlow / Keras
    *   Pandas
    *   NumPy
    *   Scikit-learn (for `MinMaxScaler`)
    *   Joblib (for saving the scaler object - optional, as parameters are saved separately for JS)
*   **Model Conversion:**
    *   `tensorflowjs_converter`
*   **Frontend / Inference:**
    *   HTML5
    *   CSS3
    *   JavaScript
    *   TensorFlow.js

## Project Structure


*(Note: The `web_anomaly_checker` directory needs to be created manually, and the specified files/folders copied or moved into it after running the Python script and the converter.)*

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Set up Python Environment:** (Recommended: Use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn joblib tensorflowjs
    ```

4.  **Prepare Data:** Ensure `invoices.csv` is present in the root directory.

5.  **Train Model & Generate Files:** Run the Python script. This will create the `.h5` model, scaler files, and threshold file.
    ```bash
    python anomaly_detector_train.py
    ```

6.  **Convert Keras Model to TF.js:**
    ```bash
    tensorflowjs_converter --input_format keras invoice_anomaly_model.h5 web_anomaly_checker/tfjs_model
    ```
    *(This command assumes you've created the `web_anomaly_checker` directory first. It saves the converted model directly into the `tfjs_model` subdirectory within it.)*

7.  **Prepare Web App Directory:** Copy the generated `scaler_params.json` and `anomaly_threshold.json` files into the `web_anomaly_checker` directory:
    ```bash
    cp scaler_params.json web_anomaly_checker/
    cp anomaly_threshold.json web_anomaly_checker/
    ```
    *(Adjust paths if your structure differs)*

## Usage

1.  **Navigate to the Web App Directory:**
    ```bash
    cd web_anomaly_checker
    ```

2.  **Start a Local Web Server:** Because browsers have security restrictions about loading files directly from the filesystem (`file://`), you need a simple web server. Python provides one built-in.
    *   For Python 3: `python -m http.server`
    *   For Python 2: `python -m SimpleHTTPServer`

3.  **Access the Web App:** Open your web browser and go to `http://localhost:8000` (or the port specified by the server, usually 8000).

4.  **Enter Invoice Data:** Fill in the details in the web form.
    *   *Note:* The date input expects `YYYY-MM-DD` format.

5.  **Check for Anomaly:** Click the "Check for Anomaly" button.

6.  **View Result:** The application will display whether the input is considered "Normal" or an "Anomaly" based on the model's reconstruction error and the predefined threshold. Check the browser's developer console (F12) for detailed logs and potential errors.

## How It Works

1.  **Autoencoder:** An unsupervised neural network is trained on the scaled numerical/date features of the invoice dataset. It learns to compress (encode) the input data into a lower-dimensional representation (latent space) and then reconstruct (decode) it back to the original dimensions.
2.  **Reconstruction Error:** The model is trained to minimize the difference (Mean Absolute Error - MAE) between the original input and its reconstruction. It becomes good at reconstructing "normal" data it saw during training.
3.  **Anomaly Threshold:** When presented with new, potentially anomalous data, the autoencoder struggles to reconstruct it accurately, resulting in a higher reconstruction error. A threshold is calculated (e.g., 99th percentile of errors on training data) to distinguish between normal reconstruction errors and potentially anomalous ones.
4.  **TensorFlow.js:** The trained Keras model is converted into a format compatible with TensorFlow.js, allowing it to run directly in the web browser without needing a backend server for inference.
5.  **Client-Side Preprocessing:** The JavaScript code meticulously preprocesses the user's input (date conversion, scaling using *exactly* the same parameters as Python) before feeding it to the TF.js model. This ensures consistency between training and inference.
6.  **Inference & Decision:** The TF.js model predicts (reconstructs) the preprocessed input. The MAE between the input and reconstruction is calculated and compared against the loaded threshold to classify the input as normal or anomalous.

## Potential Improvements

*   **Handle Categorical Features:** Incorporate text features (like `email`, `address`, `city`, `job`) using techniques like embeddings or one-hot encoding (though the latter might significantly increase input dimensions).
*   **More Sophisticated Date Features:** Extract features like day-of-week, month, is_weekend, etc., from the date instead of just using the timestamp.
*   **Alternative Models:** Explore other anomaly detection algorithms (e.g., Isolation Forest, One-Class SVM, VAEs) which might perform differently.
*   **Robust Preprocessing:** Implement more advanced NaN imputation strategies if needed.
*   **Threshold Tuning:** Adjust the percentile threshold based on the desired sensitivity and tolerance for false positives/negatives.
*   **UI/UX:** Enhance the web interface for better usability and visualization.
*   **Deployment:** Deploy the web application to a static web host (like GitHub Pages, Netlify, Vercel).

## License

(Optional: Add your preferred license, e.g., MIT)
