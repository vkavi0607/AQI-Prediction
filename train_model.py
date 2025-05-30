import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

def train_model(df, features, target='AQI'):
    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # Save model and scaler to the models directory
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)  # Create the folder if it doesn't exist

    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))

    return model, scaler, rmse
