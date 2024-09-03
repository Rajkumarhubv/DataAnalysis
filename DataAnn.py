import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Function to display dataset summary
def dataset_summary(df):
    st.write("Dataset Summary:")
    st.write(df.describe())

# Function to apply KMeans clustering
def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_

# Function to apply DBSCAN clustering
def apply_dbscan(data, eps=0.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)

# Function to apply Hierarchical clustering
def apply_hierarchical(data, n_clusters=3):
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    return hierarchical.fit_predict(data)

# Streamlit App
st.title("Clustering App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.write(df)

    # Select only columns with integer or float data types
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])
    
    if not numeric_columns.empty:
        st.write("Numeric Columns:")
        st.write(numeric_columns)

        # Display dataset summary
        dataset_summary(numeric_columns)

        # Clustering options
        clustering_method = st.selectbox("Select Clustering Algorithm", 
                                         ("K-Means", "DBSCAN", "Hierarchical"))

        # Scaling data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_columns)

        # Apply clustering based on the selected method
        if clustering_method == "K-Means":
            n_clusters = st.slider("Select Number of Clusters (K)", 2, 10, 3)
            labels = apply_kmeans(scaled_data, n_clusters)
        elif clustering_method == "DBSCAN":
            eps = st.slider("Select Epsilon (eps)", 0.1, 1.0, 0.5)
            min_samples = st.slider("Select Minimum Samples", 1, 10, 5)
            labels = apply_dbscan(scaled_data, eps, min_samples)
        elif clustering_method == "Hierarchical":
            n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)
            labels = apply_hierarchical(scaled_data, n_clusters)

        # Show clustering results
        st.write(f"Clustering Results using {clustering_method}:")
        numeric_columns['Cluster'] = labels
        st.write(numeric_columns)
    else:
        st.write("No numeric columns found in the uploaded file.")
