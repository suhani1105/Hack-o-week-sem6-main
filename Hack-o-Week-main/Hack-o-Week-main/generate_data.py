import pandas as pd
import numpy as np

# Create datetime range (7 days, hourly)
timestamps = pd.date_range(
    start="2026-01-01 00:00",
    periods=24 * 7,
    freq="H"
)

data = []
for ts in timestamps:
    hour = ts.hour

    # Base load
    energy = np.random.randint(15, 25)

    # Daytime usage
    if 8 <= hour <= 16:
        energy += np.random.randint(10, 20)

    # Evening peak
    if 18 <= hour <= 22:
        energy += np.random.randint(30, 45)

    data.append([
        ts.strftime("%Y-%m-%d %H:%M"),
        "Dorm-A",
        energy
    ])

df = pd.DataFrame(data, columns=["timestamp", "dorm_id", "energy_kWh"])
df.to_csv("dorm_electricity_data.csv", index=False)

print("✅ 7-day dorm electricity data generated successfully!")
