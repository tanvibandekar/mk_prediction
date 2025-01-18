from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np 
from flask_cors import CORS  # Import CORS

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load the trained model and scaler
model = load_model('trained_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to calculate CL and Vd from patient demographics
def calculate_cl_vd(weight, height):
    # Calculate BMI and BSA
    bmi = weight / (height / 100) ** 2
    bsa = np.sqrt((height * weight) / 3600)
    
    # Allometric scaling for CL and linear scaling for Vd
    CL = 13.2 * (weight / 70.0) ** 0.75  # Example base clearance (L/h)
    Vd = 172 * (weight / 70.0)  # Example base volume (L)
    
    return CL, Vd, bmi, bsa

# Function to predict concentration for a new patient
def predict_concentration(age, weight, height, time_point):
    # Calculate CL and Vd based on the patient's weight and height
    CL, Vd, bmi, bsa = calculate_cl_vd(weight, height)
    
    # Append calculated CL and Vd to the patient data
    patient_data_with_cl_vd = [age, weight, height, bmi, bsa, CL, Vd]
    
    # Standardize the new patient's data using the scaler
    patient_data_scaled = scaler.transform([patient_data_with_cl_vd + [time_point]])  # Add time
    
    # Predict concentration
    predicted_concentration = model.predict(patient_data_scaled)
    return predicted_concentration[0][0]

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.json
        
        # Extract the required parameters
        age = data['age']
        weight = data['weight']
        height = data['height']
        time_point = data['time_point']
        
        # Load the simulated data
        df = pd.read_csv('simulated_patient_data.csv')

        # Features (input data)
        X = df[['Age', 'Weight (kg)', 'Height (cm)', 'BMI', 'BSA (m^2)',
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
        model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test loss: {loss}')
        
        # Get predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error (MSE): {mse}')

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, y_pred)
        print(f'Mean Absolute Error (MAE): {mae}')

        # Calculate R-squared (R²)
        r2 = r2_score(y_test, y_pred)
        print(f'R-squared (R²): {r2}')

        # Optional: Visualize the results (prediction vs actual)
        plt.scatter(y_test, y_pred)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
        plt.xlabel('Actual Concentration')
        plt.ylabel('Predicted Concentration')
        plt.title('Actual vs Predicted Concentration')
        # plt.show()
                
        # Call the prediction function
        predicted_conc = predict_concentration(age, weight, height, time_point)
        
        # Return the prediction as a JSON response
        return jsonify({
            'predicted_concentration': float(predicted_conc),  # Cast to Python float
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
    # app.run(host='0.0.0.0', port=5000, debug=True)
