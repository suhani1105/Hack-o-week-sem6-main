import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Semester Forecast Dashboard", layout="wide")

st.title("📊 Semester-End Usage Forecast Dashboard")

# ===============================
# Upload Files
# ===============================
usage_file = st.file_uploader("Upload Historical Usage CSV", type=["csv"])
events_file = st.file_uploader("Upload Event Calendar CSV", type=["csv"])

semester_end_input = st.date_input("Select Semester End Date")

if usage_file and events_file:

    # ===============================
    # Load Data
    # ===============================
    usage_df = pd.read_csv(usage_file)
    events_df = pd.read_csv(events_file)

    # Convert date columns
    usage_df['date'] = pd.to_datetime(usage_df['date'])
    events_df['date'] = pd.to_datetime(events_df['date'])

    # ===============================
    # Aggregate Historical Usage (Daily)
    # ===============================
    usage_df = usage_df.groupby('date')['usage'].sum().reset_index()

    # ===============================
    # Merge with Event Calendar
    # ===============================
    df = pd.merge(usage_df, events_df, on='date', how='left')

    # Fill missing event weights with 1
    df['weight'] = df['weight'].fillna(1)

    # Adjust usage based on event impact
    df['adjusted_usage'] = df['usage'] * df['weight']

    df = df.set_index('date')

    st.subheader("📈 Historical Adjusted Usage")
    st.line_chart(df['adjusted_usage'])

    # ===============================
    # Exponential Smoothing Model
    # ===============================
    model = ExponentialSmoothing(
        df['adjusted_usage'],
        trend='add',
        seasonal='add',
        seasonal_periods=7
    )

    fit = model.fit()

    last_date = df.index.max()
    semester_end = pd.to_datetime(semester_end_input)

    days_remaining = (semester_end - last_date).days

    if days_remaining > 0:

        forecast = fit.forecast(days_remaining)

        semester_total_forecast = forecast.sum()

        # ===============================
        # Gauge Visualization
        # ===============================
        st.subheader("🎯 Semester-End Forecast Gauge")

        max_range = semester_total_forecast * 1.5

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=semester_total_forecast,
            title={'text': "Total Forecasted Usage"},
            gauge={
                'axis': {'range': [0, max_range]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_range * 0.5], 'color': "lightgreen"},
                    {'range': [max_range * 0.5, max_range * 0.8], 'color': "yellow"},
                    {'range': [max_range * 0.8, max_range], 'color': "red"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # Forecast Trend Plot
        # ===============================
        forecast_df = pd.DataFrame({
            'date': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_remaining),
            'forecast': forecast.values
        }).set_index('date')

        st.subheader("📊 Forecast Trend")
        st.line_chart(forecast_df)

    else:
        st.error("Semester end date must be after last historical date.")
