console.log("TensorFlow.js version:", tf.version.tfjs);

const MODEL_URL = './tfjs_model/model.json'; // Path relative to index.html
const SCALER_PARAMS_URL = './scaler_params.json';
const THRESHOLD_URL = './anomaly_threshold.json';

let model = null;
let scalerParams = null;
let anomalyThreshold = null;

// --- 1. Load Model, Scaler Params, and Threshold ---
async function loadResources() {
    try {
        console.log("Loading model...");
        model = await tf.loadLayersModel(MODEL_URL);
        console.log("Model loaded successfully.");

        console.log("Loading scaler parameters...");
        const scalerResponse = await fetch(SCALER_PARAMS_URL);
        if (!scalerResponse.ok) throw new Error(`HTTP error! status: ${scalerResponse.status}`);
        scalerParams = await scalerResponse.json();
        console.log("Scaler parameters loaded:", scalerParams);

         console.log("Loading threshold...");
        const thresholdResponse = await fetch(THRESHOLD_URL);
        if (!thresholdResponse.ok) throw new Error(`HTTP error! status: ${thresholdResponse.status}`);
        const thresholdData = await thresholdResponse.json();
        anomalyThreshold = thresholdData.threshold;
        console.log("Anomaly threshold loaded:", anomalyThreshold);

        // Optional: Warm up the model (run prediction once)
        // This can make subsequent predictions faster
        tf.tidy(() => {
             // Create dummy input tensor with correct shape [1, num_features]
             const dummyInput = tf.zeros([1, scalerParams.feature_names.length]);
             model.predict(dummyInput);
             console.log('Model warmed up.');
        });


    } catch (error) {
        console.error("Error loading resources:", error);
        displayResult("Error loading model/resources. Check console.", "error");
    }
}

// --- 2. Preprocessing Function ---
function preprocessInput(productId, qty, amount, invoiceDateStr) {
    if (!scalerParams) {
        console.error("Scaler parameters not loaded yet.");
        return null;
    }

    // a) Convert date string (YYYY-MM-DD from input type="date") to Unix timestamp (seconds)
    let timestamp;
    try {
         // Important: Date constructor uses local timezone by default.
         // For consistency with Python's timestamp (UTC based), explicitly handle timezones or ensure server/client timezone alignment.
         // Simplest approach if Python used local time:
         // const dateObj = new Date(invoiceDateStr); // Might be local time
         // Or specify UTC if Python used UTC:
         const dateObj = new Date(invoiceDateStr + 'T00:00:00Z'); // Treat input as UTC date
         timestamp = Math.floor(dateObj.getTime() / 1000); // getTime is ms, convert to s
         if (isNaN(timestamp)) throw new Error("Invalid date resulted in NaN timestamp");
    } catch(e) {
        console.error("Error parsing date:", e);
        displayResult("Invalid date format.", "error");
        return null;
    }


    // b) Create the input array in the *exact* order used for training
    const inputValues = [];
    const featureOrder = scalerParams.feature_names; // ['product_id', 'qty', 'amount', 'invoice_date_timestamp']

    // Very important: Ensure the order matches the training order!
    const inputMap = {
        'product_id': parseFloat(productId),
        'qty': parseFloat(qty),
        'amount': parseFloat(amount),
        'invoice_date_timestamp': timestamp
    };

    for (const featureName of featureOrder) {
        if (inputMap.hasOwnProperty(featureName)) {
            inputValues.push(inputMap[featureName]);
        } else {
            console.error(`Missing feature in inputMap: ${featureName}`);
            return null; // Or handle default value
        }
    }

    console.log("Raw input values:", inputValues);

    // c) Scale the values using the loaded scaler parameters
    const scaledValues = inputValues.map((value, i) => {
        // Formula: X_scaled = (X - min_[i]) * scale_[i]
        const scaled = (value - scalerParams.min_[i]) * scalerParams.scale_[i];
        // Clamp values to [0, 1] range as expected by sigmoid output layer
        return Math.max(0, Math.min(1, scaled));
    });

    console.log("Scaled input values:", scaledValues);

    // d) Create TensorFlow tensor [batch_size, num_features]
    return tf.tensor2d([scaledValues], [1, scaledValues.length]);
}


// --- 3. Anomaly Detection Function ---
async function detectAnomaly(event) {
    event.preventDefault(); // Prevent default form submission

    if (!model || !scalerParams || anomalyThreshold === null) {
        displayResult("Model/resources not ready yet. Please wait and try again.", "error");
        console.error("Attempted detection before resources were loaded.");
        return;
    }

    // Get form data
    const productId = document.getElementById('product_id').value;
    const qty = document.getElementById('qty').value;
    const amount = document.getElementById('amount').value;
    const invoiceDate = document.getElementById('invoice_date').value;

     // Basic validation
    if (!productId || !qty || !amount || !invoiceDate) {
        displayResult("Please fill in all fields.", "error");
        return;
    }

    // Preprocess input
    const inputTensor = preprocessInput(productId, qty, amount, invoiceDate);

    if (!inputTensor) {
        // Error already displayed by preprocessInput if needed
        return;
    }

    // Perform prediction and calculate error within tf.tidy()
    let reconstructionError = null;
    try {
        reconstructionError = tf.tidy(() => {
            const predictionTensor = model.predict(inputTensor);
            console.log("Prediction tensor (scaled):", predictionTensor.arraySync());

            // Calculate MAE loss
            // tf.losses.meanAbsoluteError expects (labels, predictions)
            const lossTensor = tf.losses.meanAbsoluteError(inputTensor, predictionTensor);
            const loss = lossTensor.dataSync()[0]; // Get the scalar value
            console.log("Reconstruction MAE:", loss);
            return loss;
        });

        // Compare with threshold
        if (reconstructionError > anomalyThreshold) {
            displayResult(`Anomaly Detected! (Error: ${reconstructionError.toFixed(6)} > Threshold: ${anomalyThreshold.toFixed(6)})`, "anomaly");
        } else {
            displayResult(`Normal (Error: ${reconstructionError.toFixed(6)} <= Threshold: ${anomalyThreshold.toFixed(6)})`, "normal");
        }
    } catch (error) {
         console.error("Error during prediction or loss calculation:", error);
         displayResult("Error during anomaly detection.", "error");
    } finally {
         // Clean up the input tensor manually if not using tf.tidy for the whole block
         if (inputTensor) inputTensor.dispose();
         console.log("Tensor disposed.");
    }
}

// --- 4. Display Result ---
function displayResult(message, type) {
    const resultDiv = document.getElementById('result');
    resultDiv.textContent = message;
    resultDiv.className = type; // Add class for styling (normal, anomaly, error)
}

// --- 5. Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    loadResources(); // Load everything when the page loads
    const form = document.getElementById('invoiceForm');
    form.addEventListener('submit', detectAnomaly);
});