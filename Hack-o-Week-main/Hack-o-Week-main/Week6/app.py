import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide")
st.title("⚡ Post-Event Electricity Forecast (LSTM)")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("electricity_hourly.csv")
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values("datetime")

# =========================
# Dashboard Filter
# =========================
day_filter = st.selectbox(
    "Select Day Type",
    df['day_type'].unique()
)

filtered_df = df[df['day_type'] == day_filter]

st.subheader("Filtered Historical Data")
st.line_chart(filtered_df.set_index("datetime")["electricity"])

# =========================
# Prepare Data for LSTM
# =========================
scaler = MinMaxScaler()
values = scaler.fit_transform(filtered_df[['electricity']])

def create_sequences(data, seq_length=3):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 3
X, y = create_sequences(values, seq_length)

if len(X) > 0:

    # =========================
    # Build LSTM Model
    # =========================
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, verbose=0)

    st.success("LSTM Model Trained")

    # =========================
    # Post-Event Prediction
    # =========================
    last_sequence = values[-seq_length:]
    last_sequence = np.reshape(last_sequence, (1, seq_length, 1))

    prediction_scaled = model.predict(last_sequence)
    prediction = scaler.inverse_transform(prediction_scaled)[0][0]

    st.metric("Predicted Next Hour Electricity", round(prediction, 2))

    # =========================
    # Forecast Plot
    # =========================
    forecast_hours = 5
    future_preds = []

    current_seq = last_sequence

    for _ in range(forecast_hours):
        pred_scaled = model.predict(current_seq)
        future_preds.append(pred_scaled[0][0])
        current_seq = np.append(current_seq[:,1:,:], [[pred_scaled]], axis=1)

    future_preds = scaler.inverse_transform(
        np.array(future_preds).reshape(-1,1)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=future_preds.flatten(),
        mode='lines+markers',
        name='Forecast'
    ))

    fig.update_layout(
        title="Next 5 Hours Forecast",
        xaxis_title="Future Hours",
        yaxis_title="Electricity"
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Not enough data for LSTM training.")
