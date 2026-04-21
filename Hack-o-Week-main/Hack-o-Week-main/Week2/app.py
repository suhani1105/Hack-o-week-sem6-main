import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="Classroom Usage Forecast", layout="centered")
st.title("🏫 Classroom Electricity Usage Forecast")

# Load dataset
df = pd.read_csv("classroom_energy.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Train ARIMAX
model = SARIMAX(
    df["energy_kwh"][:-1],
    exog=df["occupancy"][:-1],
    order=(1,1,1)
)
results = model.fit(method="powell", maxiter=200, disp=False)


forecast = results.get_forecast(
    steps=1,
    exog=df["occupancy"][-1:]
)

mean = forecast.predicted_mean.iloc[0]
ci = forecast.conf_int().iloc[0]

# Display metrics
st.metric("Next-Hour Energy Forecast (kWh)", f"{mean:.2f}")
st.write(f"**95% Confidence Interval:** {ci[0]:.2f} – {ci[1]:.2f} kWh")

# Charts
st.subheader("Historical Electricity Usage")
st.line_chart(df["energy_kwh"])

st.subheader("Occupancy (Wi-Fi Devices)")
st.line_chart(df["occupancy"])

# Insight
if df["occupancy"].iloc[-1] < 15:
    st.success("Low occupancy detected – energy optimization possible ⚡")
else:
    st.info("Normal classroom usage expected.")
