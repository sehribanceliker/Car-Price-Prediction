from flask import render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from app import app

# Load the preprocessor and model
preprocessor = joblib.load('models/preprocessor.joblib')
model = joblib.load('models/random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json

        # Convert input data to DataFrame
        input_df = pd.DataFrame(input_data)

        # Preprocess the input data
        input_encoded = preprocessor.transform(input_df)

        # Make predictions
        predictions = model.predict(input_encoded)

        # Return the predictions as JSON
        return jsonify({'predictions': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})
