import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        'PM2.5 AQI Value': 'PM2.5',
        'NO2 AQI Value': 'NO2',
        'CO AQI Value': 'CO',
        'Ozone AQI Value': 'O3',
        'AQI Value': 'AQI'
    })
    required_columns = ['PM2.5', 'NO2', 'CO', 'O3', 'AQI']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df
