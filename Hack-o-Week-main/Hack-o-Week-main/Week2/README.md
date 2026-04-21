# Classroom Usage Forecasting

This project forecasts next-hour classroom electricity consumption using
Wi-Fi based occupancy data and an ARIMAX time-series model.

## Features
- Wi-Fi device count as occupancy proxy
- ARIMAX forecasting
- 95% confidence intervals
- Streamlit dashboard

## Run Instructions
1. Generate data:
   python data/generate_data.py
2. Train & forecast:
   python model/train_arimax.py
3. Launch dashboard:
   streamlit run dashboard/app.py
