import numpy as np
from sklearn.ensemble import IsolationForest
import pickle

print("Imports successful")

# Generate sample data
normal_data = np.random.normal(loc=75, scale=5, size=100)
anomalies = np.array([120, 130, 40, 35])

data = np.concatenate([normal_data, anomalies]).reshape(-1, 1)

print("Data generated")

# Train model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(data)

print("Model trained")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")