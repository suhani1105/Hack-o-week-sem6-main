import pandas as pd
import numpy as np

np.random.seed(42)

# Hourly timestamps (5 days)
timestamps = pd.date_range(start="2026-01-01 08:00", periods=120, freq="H")

# Simulated Wi-Fi occupancy
occupancy = np.clip(np.random.normal(35, 10, len(timestamps)), 5, 60)

# Simulated energy usage (kWh)
energy_kwh = 1.5 + 0.05 * occupancy + np.random.normal(0, 0.3, len(timestamps))

df = pd.DataFrame({
    "timestamp": timestamps,
    "occupancy": occupancy,
    "energy_kwh": energy_kwh
})

df.to_csv("classroom_energy.csv", index=False)
print("Dataset saved as classroom_energy.csv")
