from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
from flask_cors import CORS
import numpy as np

# Load the trained model and scaler
model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS

# Function to calculate CL and Vd from patient demographics
def calculate_cl_vd(weight, height):
    bmi = weight / (height / 100) ** 2
    bsa = np.sqrt((height * weight) / 3600)
    CL = 13.2 * (weight / 70.0) ** 0.75  # Example base clearance (L/h)
    Vd = 172 * (weight / 70.0)  # Example base volume (L)
    return CL, Vd, bmi, bsa

# Function to predict concentration for a new patient
def predict_concentration(age, weight, height, time_point):
    CL, Vd, bmi, bsa = calculate_cl_vd(weight, height)
    patient_data_with_cl_vd = [age, weight, height, bmi, bsa, CL, Vd]
    patient_data_scaled = scaler.transform([patient_data_with_cl_vd + [time_point]])
    predicted_concentration = model.predict(patient_data_scaled)
    return predicted_concentration[0][0]

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.json
        age = float(data["age"])
        weight = float(data["weight"])
        height = float(data["height"])
        time_point = float(data["time_point"])

        # Predict concentration
        predicted_conc = predict_concentration(age, weight, height, time_point)

        # Return the prediction as a JSON response
        return jsonify({
            'predicted_concentration': float(predicted_conc),
            'message': 'Prediction successful'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during prediction'
        }), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
