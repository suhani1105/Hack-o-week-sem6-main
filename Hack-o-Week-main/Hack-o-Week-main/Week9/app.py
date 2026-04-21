import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from prophet import Prophet
import matplotlib.pyplot as plt

# -----------------------------
# Create synthetic time-series data
# -----------------------------
dates = pd.date_range(start="2024-01-01", periods=120)
usage = np.random.randint(50, 200, size=120)

df = pd.DataFrame({
    "ds": dates,
    "y": usage
})

# -----------------------------
# Create usage categories
# -----------------------------
def categorize(val):
    if val < 80:
        return 0   # Low
    elif val < 140:
        return 1   # Medium
    else:
        return 2   # High

df["category"] = df["y"].apply(categorize)

# -----------------------------
# Train Naive Bayes classifier
# -----------------------------
X = df["y"].values.reshape(-1,1)
y = df["category"]

nb = GaussianNB()
nb.fit(X, y)

df["predicted_category"] = nb.predict(X)

print("\nSample classification output:\n")
print(df.head())

# -----------------------------
# Prophet Forecast
# -----------------------------
model = Prophet()
model.fit(df[["ds","y"]])

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# -----------------------------
# Plot results
# -----------------------------
plt.figure()
plt.plot(df["ds"], df["y"], label="Actual Usage")
plt.plot(forecast["ds"], forecast["yhat"], label="Forecast")
plt.xlabel("Date")
plt.ylabel("Usage")
plt.title("Time-Series Usage Forecast")
plt.legend()
plt.show()

# -----------------------------
# Simple anomaly alert
# -----------------------------
threshold = df["y"].mean() + 2*df["y"].std()

for i,val in enumerate(df["y"]):
    if val > threshold:
        print(f"ALERT: Anomaly detected on {df['ds'][i]} with value {val}")