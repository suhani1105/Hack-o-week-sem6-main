import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import websockets
import threading
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(layout="wide")
st.title("🌤 Lunch Hour Surge Prediction Dashboard")

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("weather_usage.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

# ===============================
# Train Linear Regression Model
# ===============================
X = df[['temperature', 'humidity', 'hour']]
y = df['usage']

model = LinearRegression()
model.fit(X, y)

st.success("Model Trained Successfully")

# ===============================
# Prediction Section
# ===============================
st.subheader("🔮 Predict Lunch Hour Surge")

temp_input = st.slider("Temperature (°C)", 15, 40, 28)
humidity_input = st.slider("Humidity (%)", 30, 90, 60)
hour_input = st.slider("Hour of Day", 10, 15, 13)

input_data = np.array([[temp_input, humidity_input, hour_input]])
prediction = model.predict(input_data)[0]

st.metric("Predicted Usage", round(prediction, 2))

# ===============================
# Real-Time WebSocket Simulation
# ===============================

st.subheader("📈 Real-Time Usage Monitoring")

chart_placeholder = st.empty()

usage_data = []

async def fake_websocket_stream():
    global usage_data
    while True:
        simulated_usage = prediction + np.random.normal(0, 10)
        usage_data.append(simulated_usage)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=usage_data,
            mode='lines',
            name='Real-Time Usage'
        ))

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Usage",
            height=400
        )

        chart_placeholder.plotly_chart(fig, use_container_width=True)
        await asyncio.sleep(2)

def run_async_loop():
    asyncio.new_event_loop().run_until_complete(fake_websocket_stream())

if st.button("Start Real-Time Monitoring"):
    thread = threading.Thread(target=run_async_loop)
    thread.start()
