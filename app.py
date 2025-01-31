from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, static_folder='static', template_folder='static')

# Load models
model_cases = pickle.load(open("model_cases.pkl", "rb"))
model_cfr = pickle.load(open("model_cfr.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        lat = float(data['Lat'])
        long = float(data['Long'])

        # Make predictions
        cases_prediction = model_cases.predict([[lat, long]])[0]
        cfr_prediction = model_cfr.predict([[lat, long]])[0]

        return jsonify({
            'confirmed_cases': round(cases_prediction, 2),
            'cfr': round(cfr_prediction, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
