import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simulated sensor vehicle count data (per minute)
vehicle_counts = [10, 12, 15, 18, 40, 22, 25, 27, 30, 35]
time = np.arange(len(vehicle_counts))

# Prepare polynomial regression
X = time.reshape(-1, 1)
y = np.array(vehicle_counts)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

# Predict future traffic
future_time = np.arange(len(vehicle_counts) + 5).reshape(-1, 1)
future_poly = poly.transform(future_time)
predictions = model.predict(future_poly)

# Detect anomalies (simple threshold method)
threshold = np.mean(vehicle_counts) + 2 * np.std(vehicle_counts)
anomalies = [i for i, v in enumerate(vehicle_counts) if v > threshold]

# Real-time style bar chart
plt.ion()
fig, ax = plt.subplots()

for i, count in enumerate(vehicle_counts):
    ax.clear()
    ax.bar(range(len(vehicle_counts)), vehicle_counts)
    ax.set_title("Real-time Vehicle Count")
    ax.set_xlabel("Time Interval")
    ax.set_ylabel("Vehicle Count")

    if i in anomalies:
        print(f"ALERT: Traffic anomaly detected at time {i} with {vehicle_counts[i]} vehicles")

    plt.pause(0.8)

plt.ioff()

# Plot regression prediction
plt.figure()
plt.scatter(time, vehicle_counts)
plt.plot(future_time, predictions)
plt.title("Polynomial Regression Traffic Prediction")
plt.xlabel("Time")
plt.ylabel("Vehicle Count")
plt.show()