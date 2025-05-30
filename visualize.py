import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def show_gauge(prediction):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "AQI Prediction"},
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
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    fig.show()

def show_feature_importance(features, importances):
    df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fig = px.bar(df.sort_values(by='Importance'), x='Importance', y='Feature', title="Feature Importances")
    fig.show()
