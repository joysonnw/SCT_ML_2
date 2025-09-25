import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Retail Customer Segmentation using KMeans")

@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Select Features for Clustering")
features = st.multiselect(
    "Choose features:",
    options=df.columns.tolist(),
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

if len(features) < 2:
    st.warning("Please select at least 2 features for visualization and clustering.")
else:
    X = df[features]

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=5)

    
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    
    df["Cluster"] = clusters

    st.subheader("Clustered Data Sample")
    st.dataframe(df.head())

    
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=clusters,
        cmap="viridis"
    )
    
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c="red", s=200, alpha=0.7, marker="X")

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Customer Clusters")
    st.pyplot(fig)
