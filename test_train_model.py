import sys
import os
import pandas as pd

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_model import train_model

def test_train_model():
    data = pd.DataFrame({
        'PM2.5': [10, 20, 30, 40, 50],
        'NO2': [5, 10, 15, 20, 25],
        'CO': [0.5, 1.0, 1.5, 2.0, 2.5],
        'O3': [25, 50, 75, 100, 125],
        'AQI': [50, 100, 150, 200, 250]
    })
    features = ['PM2.5', 'NO2', 'CO', 'O3']
    model, scaler, rmse = train_model(data, features)

    # Basic assertions
    assert model is not None
    assert scaler is not None
    assert rmse >= 0
