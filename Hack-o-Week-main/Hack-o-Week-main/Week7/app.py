import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📊 Usage Clustering & Savings Forecast Dashboard")

# ============================
# Load Data
# ============================
df = pd.read_csv("usage_profiles.csv")

X = df[['avg_usage', 'peak_usage', 'offpeak_usage']]

# ============================
# Apply K-Means Clustering
# ============================
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

st.subheader("Clustered Usage Profiles")
st.dataframe(df)

# ============================
# Cluster-wise Regression Forecast
# ============================
st.subheader("📈 Cluster-wise Forecasting")

forecast_results = []
savings_results = []

for cluster_id in df['cluster'].unique():

    cluster_data = df[df['cluster'] == cluster_id]

    X_cluster = cluster_data[['avg_usage', 'offpeak_usage']]
    y_cluster = cluster_data['peak_usage']

    model = LinearRegression()
    model.fit(X_cluster, y_cluster)

    # Predict optimized peak usage
    predicted_peak = model.predict(X_cluster)

    cluster_data = cluster_data.copy()
    cluster_data['predicted_peak'] = predicted_peak

    # Calculate potential savings (if peak reduced by 10%)
    cluster_data['optimized_peak'] = cluster_data['peak_usage'] * 0.9
    cluster_data['savings'] = cluster_data['peak_usage'] - cluster_data['optimized_peak']

    forecast_results.append(cluster_data)

    total_savings = cluster_data['savings'].sum()
    savings_results.append({
        "Cluster": f"Cluster {cluster_id}",
        "Savings": total_savings
    })

forecast_df = pd.concat(forecast_results)

st.dataframe(forecast_df)

# ============================
# Pie Chart: Savings Potential
# ============================
st.subheader("💰 Savings Potential by Cluster")

savings_df = pd.DataFrame(savings_results)

fig = px.pie(
    savings_df,
    names="Cluster",
    values="Savings",
    title="Savings Distribution Across Clusters"
)

st.plotly_chart(fig, use_container_width=True)
