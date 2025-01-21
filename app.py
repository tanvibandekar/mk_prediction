from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from flask_cors import CORS

# Load the trained model and scaler
model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

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
    patient_data_scaled = scaler.transform([patient_data_with_cl_vd + [time_point]])  # Add time
    predicted_concentration = model.predict(patient_data_scaled)
    return predicted_concentration[0][0]

# HTML template with the form
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Patient Prediction</title>
</head>
<body>
    <h1>Patient Drug Concentration Prediction</h1>
    <form action="/predict" method="post">
        <label for="age">Age (years):</label>
        <input type="number" id="age" name="age" step="0.1" required><br><br>
        <label for="weight">Weight (kg):</label>
        <input type="number" id="weight" name="weight" step="0.1" required><br><br>
        <label for="height">Height (cm):</label>
        <input type="number" id="height" name="height" step="0.1" required><br><br>
        <label for="time_point">Time (hours):</label>
        <input type="number" id="time_point" name="time_point" step="0.1" required><br><br>
        <button type="submit">Predict</button>
    </form>
    <p id="result">{{ result }}</p>
</body>
</html>
"""

# Home route to render the form
@app.route('/')
def index():
    return render_template_string(form_html, result="")

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        age = float(request.form["age"])
        weight = float(request.form["weight"])
        height = float(request.form["height"])
        time_point = float(request.form["time_point"])
        
        # Predict concentration
        predicted_conc = predict_concentration(age, weight, height, time_point)
        
        # Return the form with the result
        return render_template_string(form_html, result=f"Predicted concentration: {predicted_conc:.2f} mg/L")
    
    except Exception as e:
        return render_template_string(form_html, result=f"An error occurred: {str(e)}")

# Run the Flask app
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, debug=True)
