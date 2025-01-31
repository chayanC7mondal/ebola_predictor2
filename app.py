import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="static")

# Load pre-trained models
model_cases = pickle.load(open("model_cases.pkl", "rb"))
model_cfr = pickle.load(open("model_cfr.pkl", "rb"))

@app.route("/")
def index():
    """Render the frontend."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle file upload and return predictions."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Check if the file is a CSV
    if not file.filename.endswith(".csv"):
        return jsonify({"error": "Only CSV files are allowed"}), 400

    # Read the uploaded file
    try:
        test_data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to read CSV: {str(e)}"}), 400

    # Ensure required columns are present
    required_columns = ["Lat", "Long_"]
    if not all(col in test_data.columns for col in required_columns):
        return jsonify({"error": f"CSV must contain columns: {', '.join(required_columns)}"}), 400

    # Make predictions
    try:
        X_test = test_data[required_columns]
        predicted_cases = model_cases.predict(X_test)
        predicted_cfr = model_cfr.predict(X_test)

        # Add predictions to the dataframe
        test_data["Predicted_CFR"] = predicted_cfr
        test_data["Predicted_Confirmed_Cases"] = predicted_cases

        # Convert results to a list of dictionaries
        results = test_data.to_dict(orient="records")
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Failed to make predictions: {str(e)}"}), 500

if __name__ == "__main__":
    # Ensure models exist
    if not os.path.exists("model_cases.pkl") or not os.path.exists("model_cfr.pkl"):
        print("Error: Model files not found. Make sure 'model_cases.pkl' and 'model_cfr.pkl' are present.")
        exit(1)

    # Run Flask app
    app.run(debug=True)
