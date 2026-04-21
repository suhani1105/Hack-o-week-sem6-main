import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load data
df = pd.read_csv("classroom_energy.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Train-test split
train = df.iloc[:-1]
test_exog = df.iloc[-1:]["occupancy"]

# ARIMAX Model
model = SARIMAX(
    train["energy_kwh"],
    exog=train["occupancy"],
    order=(1, 1, 1),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(method="powell", maxiter=200, disp=False)


# Forecast next hour
forecast = results.get_forecast(steps=1, exog=test_exog)

mean = forecast.predicted_mean.iloc[0]
ci = forecast.conf_int().iloc[0]

print("Next-Hour Forecast")
print("------------------")
print(f"Predicted Energy: {mean:.2f} kWh")
print(f"95% CI: {ci[0]:.2f} – {ci[1]:.2f} kWh")
