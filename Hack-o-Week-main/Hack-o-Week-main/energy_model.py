import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import plotly.express as px

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv("dorm_electricity_data.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(by='timestamp')

# -------------------------------
# Moving Average
# -------------------------------
df['smoothed_kWh'] = df['energy_kWh'].rolling(window=3).mean()

# -------------------------------
# Evening Hours
# -------------------------------
df['hour'] = df['timestamp'].dt.hour
evening_df = df.loc[(df['hour'] >= 18) & (df['hour'] <= 23)].copy()

# -------------------------------
# Feature Engineering (SAFE)
# -------------------------------
evening_df['prev_hour_usage'] = evening_df['energy_kWh'].shift(1)
evening_df['rolling_avg'] = evening_df['energy_kWh'].rolling(window=3).mean()

evening_df.dropna(inplace=True)

# -------------------------------
# Model
# -------------------------------
X = evening_df[['prev_hour_usage', 'rolling_avg']]
y = evening_df['energy_kWh']

model = LinearRegression()
model.fit(X, y)

evening_df['predicted_peak'] = model.predict(X)

# -------------------------------
# Evaluation
# -------------------------------
mae = mean_absolute_error(y, evening_df['predicted_peak'])
print("Mean Absolute Error:", round(mae, 2))

# -------------------------------
# Visualization
# -------------------------------
fig1 = px.line(
    df,
    x='timestamp',
    y=['energy_kWh', 'smoothed_kWh'],
    title='Electricity Usage: Raw vs Smoothed'
)
fig1.show()

fig2 = px.line(
    evening_df,
    x='timestamp',
    y=['energy_kWh', 'predicted_peak'],
    title='Evening Peak Prediction'
)
fig2.show()
