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
# form_html = """
# <!DOCTYPE html>
# <html lang="en">

# <head>
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Patient Prediction</title>
#     <script>

#         async function predictConcentration(event) {
#             event.preventDefault(); // Prevent form submission from reloading the page

#             const age = parseFloat(document.getElementById("age").value);
#             const weight = parseFloat(document.getElementById("weight").value);
#             const height = parseFloat(document.getElementById("height").value);
#             const timePoint = parseFloat(document.getElementById("time_point").value);

#             const requestData = {
#                 age: age,
#                 weight: weight,
#                 height: height,
#                 time_point: timePoint
#             };

#             try {
#                 const response = await fetch("http://127.0.0.1:5000/predict", {
#                     method: "POST",
#                     headers: {
#                         "Content-Type": "application/json"
#                     },
#                     body: JSON.stringify(requestData)
#                 });

#                 if (response.ok) {
#                     const result = await response.json();
#                     document.getElementById("result").innerText =
#                         `Predicted concentration: ${result.predicted_concentration.toFixed(2)} mg/L`;
#                 } else {
#                     const errorData = await response.json();
#                     document.getElementById("result").innerText =
#                         `Error: ${errorData.error || "Unknown error occurred"}`;
#                 }
#             } catch (error) {
#                 document.getElementById("result").innerText = `Error: ${error.message}`;
#             }
#         }


#     </script>
# </head>

# <body>
#     <h1>Predict Imatinib Serum Concentration</h1>
#     <form id="predictionForm" onsubmit="predictConcentration(event)">
#         <label for="age">Age (years):</label>
#         <input type="number" id="age" name="age" step="0.1" required><br><br>

#         <label for="weight">Weight (kg):</label>
#         <input type="number" id="weight" name="weight" step="0.1" required><br><br>

#         <label for="height">Height (cm):</label>
#         <input type="number" id="height" name="height" step="0.1" required><br><br>

#         <label for="time_point">Time (hours):</label>
#         <input type="number" id="time_point" name="time_point" step="0.1" required><br><br>

#         <button type="submit">Predict</button>
#     </form>
#     <p id="result"></p>
# </body>

# </html>
# """

form_html = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Patient Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #D4F1F4;
            /* Replace with your image URL */
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            color: #333;
        }

        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: rgba(0, 123, 255, 0.9);
            /* Semi-transparent navbar */
            padding: 10px 20px;
            color: #fff;
        }

        .navbar img {
            height: 50px;
            border-radius: 50px;
        }

        .navbar h1 {
            margin: 0;
            font-size: 1.5rem;
            text-align: center;
        }

        .scroll-p {
            padding: 20px;
            color: black;
            font-weight: bold;
            text-align: center;
        }

        .container {
            max-width: 600px;
            margin: 40px auto;
            background: rgba(255, 255, 255, 0.8);
            /* Transparent white background for the form */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }

        .container h2 {
            text-align: center;
            padding-bottom: 10px;
        }

        form {
            display: flex;
            flex-direction: column;
        }

        label {
            margin-bottom: 5px;
            font-weight: bold;
        }

        input {
            margin-bottom: 15px;
            padding: 10px;
            font-size: 1rem;
            border: 1px solid #ccc;
            border-radius: 5px;
        }

        button {
            padding: 10px;
            font-size: 1rem;
            color: #fff;
            background-color: #007bff;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        button:hover {
            background-color: #0056b3;
        }

        #result {
            margin-top: 15px;
            font-weight: bold;
            color: #333;
        }

        .contact {
            position: absolute;
            top: 105%;
            /* Centers vertically */
            right: 20%;
            /* Adjusts horizontal positioning */
            transform: translateY(-50%);
            /* Centers the element properly */
            padding: 15px;
            border-radius: 8px;
            text-align: right;
            font-family: Arial, sans-serif;
            max-width: 250px;
            /* Ensures the box is compact */
        }

        .contact h3 {
            color: #007bff;
            margin-bottom: 10px;
            font-size: 1.25rem;
        }

        .contact a {
            color: #0056b3;
            text-decoration: none;
        }

        .contact a:hover {
            text-decoration: underline;
        }

    </style>
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
        
            // Dynamically set the backend URL based on deployment environment
            const backendURL = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
                ? "http://127.0.0.1:8000/predict"  // Local development
                : "https://dose-prediction.onrender.com";  // Deployed URL (replace with your actual domain)
        
            try {
                const response = await fetch(backendURL, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(requestData)
                });
        
                if (response.ok) {
                    const result = await response.json();
                    const predictedConcentration = parseFloat(result.predicted_concentration).toFixed(2);
                    document.getElementById("result").innerText =
                        `Predicted concentration: ${predictedConcentration} mg/L`;
                } else {
                    const errorData = await response.text(); // Use .text() if response is not JSON
                    document.getElementById("result").innerText =
                        `Error: ${errorData || "Unknown error occurred"}`;
                }
            } catch (error) {
                document.getElementById("result").innerText = `Error: ${error.message}`;
            }
        }

    </script>
</head>

<body>
    <div class="navbar">
        <h1>Imatinib Concentration Predictor</h1>
    </div>
    <div class="scroll-p">
        <marquee width="800">Welcome to ImatiPredict!
            We are excited to support your journey in delivering personalized patient care. ImatiPredict empowers you
            with precise serum concentration predictions for Imatinib therapy, enabling confident and informed dosing
            decisions. Designed to enhance treatment outcomes through advanced modeling and AI-driven insights, our
            platform is your partner in elevating the standards of precision medicine and patient management.
        </marquee>
    </div>
    <div class="container-wrapper">
        <div class="container">
            <h2>Predict Imatinib Serum Concentration</h2>
            <form id="predictionForm" onsubmit="predictConcentration(event)">
                <label for="age">Age (years):</label>
                <input type="number" id="age" name="age" step="0.1" required>

                <label for="weight">Weight (kg):</label>
                <input type="number" id="weight" name="weight" step="0.1" required>

                <label for="height">Height (cm):</label>
                <input type="number" id="height" name="height" step="0.1" required>

                <label for="time_point">Time (hours):</label>
                <input type="number" id="time_point" name="time_point" step="0.1" required>

                <button type="submit">Predict</button>
            </form>
            <p id="result"></p>
        </div>
        <div class="contact">
            <h3>Contact Information</h3>
            <p>
                <strong>Dr.Muthukumar M</strong> <br>
                Research Scholar <br>
                PES University <br>
                <a href="mailto:Muthukumar@pes.edu">Muthukumar@pes.edu</a>
            </p>
        </div>

    </div>

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
        
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from tensorflow import keras
        from tensorflow.keras import layers
        # Load the simulated data
        df = pd.read_csv('simulated_patient_data.csv')

        # Features (input data)
        X = df[['Age (years)', 'Weight (kg)', 'Height (cm)', 'BMI', 'BSA (m^2)',
                'Clearance (L/h)', 'Vd (L)', 'Time (hours)']].values

        # Target variable (serum concentration)
        y = df['Concentration (mg/L)'].values

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build the neural network model with dropout and additional tuning
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
            layers.Dropout(0.2),  # Add dropout for regularization
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),  # Add dropout for regularization
            layers.Dense(32, activation='relu'),
            layers.Dense(1)  # Single output for serum concentration
        ])

        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Predict concentration
        predicted_conc = predict_concentration(age, weight, height, time_point)

        # Convert numpy.float32 to Python float for JSON serialization
        print("predicted concentration: ", float(predicted_conc))
        return jsonify({"predicted_concentration": float(predicted_conc)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=5000)
