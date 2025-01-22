from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
import joblib
import numpy as np
from flask_cors import CORS

# Load the trained model and scaler
model = load_model("trained_model.h5")
scaler = joblib.load("scaler.pkl")

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
    patient_data_scaled = scaler.transform(
        [patient_data_with_cl_vd + [time_point]]
    )  # Add time
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
    <script>

        async function predictConcentration(event) {
            event.preventDefault(); // Prevent form submission from reloading the page

            const age = parseFloat(document.getElementById("age").value);
            const weight = parseFloat(document.getElementById("weight").value);
            const height = parseFloat(document.getElementById("height").value);
            const timePoint = parseFloat(document.getElementById("time_point").value);

            const requestData = {
                age: age,
                weight: weight,
                height: height,
                time_point: timePoint
            };

            try {
                const response = await fetch("http://127.0.0.1:5000/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(requestData)
                });

                if (response.ok) {
                    const result = await response.json();
                    document.getElementById("result").innerText =
                        `Predicted concentration: ${result.predicted_concentration.toFixed(2)} mg/L`;
                } else {
                    const errorData = await response.json();
                    document.getElementById("result").innerText =
                        `Error: ${errorData.error || "Unknown error occurred"}`;
                }
            } catch (error) {
                document.getElementById("result").innerText = `Error: ${error.message}`;
            }
        }


    </script>
</head>

<body>
    <h1>Patient Drug Concentration Prediction</h1>
    <form id="predictionForm" onsubmit="predictConcentration(event)">
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
    <p id="result"></p>
</body>

</html>
"""


# Home route to render the form
@app.route("/")
def index():
    return render_template_string(form_html, result="")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect data from the JSON body
        data = request.json
        age = float(data["age"])
        weight = float(data["weight"])
        height = float(data["height"])
        time_point = float(data["time_point"])

        # Predict concentration
        predicted_conc = predict_concentration(age, weight, height, time_point)

        # Convert numpy.float32 to Python float for JSON serialization
        return jsonify({"predicted_concentration": float(predicted_conc)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
