import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(page_title="AQI Predictor", layout="wide")
st.title("🌫️ Air Quality Index (AQI) Predictor")

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'global_air_pollution_dataset.csv')
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        'PM2.5 AQI Value': 'PM2.5',
        'NO2 AQI Value': 'NO2',
        'CO AQI Value': 'CO',
        'Ozone AQI Value': 'O3',
        'AQI Value': 'AQI'
    })
    return df

data = load_data()
features = ['PM2.5', 'NO2', 'CO', 'O3']
X = data[features]
y = data['AQI']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Sidebar inputs
st.sidebar.header("🛠 Input Features")
pm25 = st.sidebar.slider("PM2.5", 0.0, 500.0, 55.0)
no2 = st.sidebar.slider("NO2", 0.0, 100.0, 20.0)
co = st.sidebar.slider("CO", 0.0, 50.0, 1.5)
o3 = st.sidebar.slider("O3", 0.0, 200.0, 40.0)

input_df = pd.DataFrame({'PM2.5': [pm25], 'NO2': [no2], 'CO': [co], 'O3': [o3]})
input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)[0]

# AQI Gauge
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={'text': "Predicted AQI"},
    gauge={
        'axis': {'range': [0, 500]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 50], 'color': "green"},
            {'range': [50, 100], 'color': "yellow"},
            {'range': [100, 150], 'color': "orange"},
            {'range': [150, 200], 'color': "red"},
            {'range': [200, 300], 'color': "purple"},
            {'range': [300, 500], 'color': "maroon"}
        ]
    }
))
st.plotly_chart(fig, use_container_width=True)

# AQI Category
st.subheader("AQI Category")
if prediction <= 50:
    st.success("Good: Air quality is satisfactory.")
elif prediction <= 100:
    st.info("Moderate: Air quality is acceptable.")
elif prediction <= 150:
    st.warning("Unhealthy for Sensitive Groups.")
elif prediction <= 200:
    st.error("Unhealthy: Health effects may occur.")
elif prediction <= 300:
    st.error("Very Unhealthy: Health alert.")
else:
    st.error("Hazardous: Emergency conditions.")

# AQI Distribution
st.subheader("📊 AQI Distribution")
fig_hist = px.histogram(data, x="AQI", nbins=50, title="Distribution of AQI Values")
st.plotly_chart(fig_hist, use_container_width=True)

# Correlation Heatmap
st.subheader("🔗 Correlation Between Pollutants")
corr = data[features + ['AQI']].corr()
fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
st.plotly_chart(fig_corr, use_container_width=True)

# Scatter plots with user input highlighted
st.subheader("📌 AQI vs Individual Pollutants (with Your Input)")
for col in features:
    fig_scatter = px.scatter(data, x=col, y='AQI', title=f"AQI vs {col}")
    fig_scatter.add_trace(
        go.Scatter(x=[input_df[col][0]], y=[prediction], mode='markers+text',
                   marker=dict(color='red', size=12),
                   name='Your Input',
                   text=["Your Input"], textposition="top center")
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Feature Importance
st.subheader("🔍 Feature Importance")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True)
fig_imp = px.bar(importance, x='Importance', y='Feature', title="Feature Importance")
st.plotly_chart(fig_imp, use_container_width=True)

# Health Recommendations Table
st.subheader("🏥 Health Recommendations by AQI Category")
health_tips = pd.DataFrame({
    "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500"],
    "Category": ["Good", "Moderate", "Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"],
    "Health Advice": [
        "Air quality is satisfactory. Enjoy outdoor activities.",
        "Air quality is acceptable. Some pollutants may affect very few sensitive individuals.",
        "Sensitive groups (asthma, elderly) should limit prolonged outdoor exertion.",
        "Everyone may begin to experience health effects; sensitive groups may experience more serious effects.",
        "Health alert: everyone may experience more serious health effects.",
        "Emergency conditions. Entire population is more likely to be affected."
    ]
})
st.dataframe(health_tips)

# Raw Data Preview
st.subheader("📄 Sample Raw Data")
st.dataframe(data.head())
