import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

st.set_page_config(layout="wide")
st.title("❄ Cooling Needs Prediction Dashboard")

# ============================
# Load Dataset
# ============================
df = pd.read_csv("cooling_data.csv")

# Encode categorical variables
zone_encoder = LabelEncoder()
cooling_encoder = LabelEncoder()

df['zone_encoded'] = zone_encoder.fit_transform(df['zone'])
df['cooling_encoded'] = cooling_encoder.fit_transform(df['cooling_need'])

X = df[['zone_encoded', 'occupancy', 'temperature']]
y = df['cooling_encoded']

# ============================
# Train Decision Tree
# ============================
model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)

st.success("Decision Tree Model Trained Successfully")

# ============================
# Prediction Section
# ============================
st.subheader("🔮 Predict Cooling Need")

zone_input = st.selectbox("Select Zone", df['zone'].unique())
occupancy_input = st.slider("Occupancy", 0, 60, 20)
temp_input = st.slider("Temperature (°C)", 15, 40, 28)

zone_encoded = zone_encoder.transform([zone_input])[0]

input_data = np.array([[zone_encoded, occupancy_input, temp_input]])
prediction_encoded = model.predict(input_data)[0]
prediction_label = cooling_encoder.inverse_transform([prediction_encoded])[0]

st.metric("Predicted Cooling Need", prediction_label)

# ============================
# Zone-wise Heatmap
# ============================
st.subheader("🔥 Zone-wise Cooling Heatmap")

# Generate simulated predictions for heatmap grid
heatmap_data = []

for zone in df['zone'].unique():
    for temp in range(22, 36, 2):
        occ = 30
        zone_enc = zone_encoder.transform([zone])[0]
        pred = model.predict([[zone_enc, occ, temp]])[0]
        heatmap_data.append({
            "Zone": zone,
            "Temperature": temp,
            "Cooling_Level": cooling_encoder.inverse_transform([pred])[0]
        })

heatmap_df = pd.DataFrame(heatmap_data)

# Convert category to numeric for heatmap intensity
level_map = {"Low": 1, "Medium": 2, "High": 3}
heatmap_df['Cooling_Score'] = heatmap_df['Cooling_Level'].map(level_map)

fig = px.density_heatmap(
    heatmap_df,
    x="Temperature",
    y="Zone",
    z="Cooling_Score",
    color_continuous_scale="RdYlBu_r"
)

st.plotly_chart(fig, use_container_width=True)
