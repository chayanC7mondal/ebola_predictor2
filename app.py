from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved models
model_cases = joblib.load('confirmed_cases_model.pkl')
model_cfr = joblib.load('cfr_model.pkl')

@app.route('/predict_cases', methods=['POST'])
def predict_cases():
    data = request.get_json()  # Get data from the frontend
    lat = data['lat']
    long = data['long']
    
    # Prepare the data for prediction
    features = np.array([[lat, long]])
    
    # Predict using the trained model
    prediction = model_cases.predict(features)
    
    return jsonify({'predicted_cases': prediction[0]})

@app.route('/predict_cfr', methods=['POST'])
def predict_cfr():
    data = request.get_json()  # Get data from the frontend
    lat = data['lat']
    long = data['long']
    
    # Prepare the data for prediction
    features = np.array([[lat, long]])
    
    # Predict using the trained model
    prediction = model_cfr.predict(features)
    
    return jsonify({'predicted_cfr': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
