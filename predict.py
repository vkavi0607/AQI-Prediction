import os
import joblib
import pandas as pd

def predict_aqi(input_data):
    # Resolve paths relative to this file
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '..', 'models', 'random_forest_model.pkl')
    scaler_path = os.path.join(base_dir, '..', 'models', 'scaler.pkl')

    # Load model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Ensure input_data is a DataFrame (if a list or array is passed)
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data], columns=['PM2.5', 'NO2', 'CO', 'O3'])

    # Scale and predict
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    return prediction
