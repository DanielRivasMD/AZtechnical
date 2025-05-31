####################################################################################################

# Install required packages
# !pip install pandas numpy scikit-learn pecanpy

####################################################################################################

# Imports
import pandas as pd
import numpy as np
from pecanpy.graph import AdjlstGraph
from pecanpy import pecanpy as node2vec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

####################################################################################################

# Load the data
df = pd.read_parquet('omics_embedding_take_home_2025.parquet')
df.head()

# Extract unique node IDs and edges
node_ids = pd.unique(df[['source_id', 'target_id']].values.ravel())
edges_iter = df[['source_id', 'target_id']].itertuples(index=False, name=None)

####################################################################################################

# Initialize graph
graph = AdjlstGraph()

# Add nodes
for node_id in node_ids:
    graph.add_node(node_id)

# Add edges
for source, target in edges_iter:
    graph.add_edge(source, target, directed=False)


####################################################################################################

### 1. Select the Right Parameters for Node2Vec

# Experiment with the following Node2Vec parameters to improve embedding quality. Use external resources as needed to understand their impact:

# - **dim**
# - **num_walks**
# - **walk_length**

####################################################################################################

# Params for initializing the model (graph traversal)
init_params = {
    "p": 1.0,
    "q": 1.0,
    "extend": False,
    "directed": False,
    "verbose": True,
}

adjust_dim = 128
adjust_num_walks = 10
adjust_walk_length = 80

# Params for embedding
# Modify the ADJUST_PARAM values in embed_params based on your experimentation
embed_params = {
    "dim": adjust_dim,
    "num_walks": adjust_num_walks,
    "walk_length": adjust_walk_length,
    "window_size": 5,
    "epochs": 5,
    "verbose": True,
}

####################################################################################################

# Initialize the model
model = node2vec.PreComp(init_params)

# Load the graph
model = model.from_adjlst_graph(adjlst_graph=graph)

# Train embedding with embedding-related params
node_embeddings = model.embed(**embed_params)

# Extract patient IDs from both sides of edges
patients = set(df[df["source_type"] == "Patient"]["source_id"]) | \
           set(df[df["target_type"] == "Patient"]["target_id"])

# Filter embeddings for patient nodes only
node_ids_array = np.array(node_ids)
patient_mask = np.isin(node_ids_array, list(patients))
patient_embeddings = node_embeddings[patient_mask]
patient_ids = node_ids_array[patient_mask]

####################################################################################################

# PCA Visualization
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(patient_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], 
            s=10, color='blue', alpha=0.7)
plt.title("PCA Projection of Patient Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

# t-SNE Visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(patient_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], 
            s=10, color='red', alpha=0.7)
plt.title("t-SNE Projection of Patient Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
plt.show()

####################################################################################################

### 2. Experiment with DBSCAN

# Please experiment with **DBSCAN** and compare its results with **KMeans**. Try to explore the differences between these two clustering algorithms.

####################################################################################################

from sklearn.cluster import DBSCAN

# DBSCAN parameters can have a significant impact. Here eps is the maximum distance for points
# to be considered neighbors. min_samples is the number of points needed to form a dense region.
dbscan = DBSCAN(eps=0.5, min_samples=5)
patient_labels_dbscan = dbscan.fit_predict(patient_embeddings)

####################################################################################################

# Fit NearestNeighbors on patient_embeddings
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(patient_embeddings)
distances, indices = neighbors_fit.kneighbors(patient_embeddings)
k_distances = distances[:, min_samples - 1]

# Sort these distances to visually inspect the k-distance plot.
k_distances_sorted = np.sort(k_distances)

# Plot the k-distance graph
plt.figure(figsize=(8, 6))
plt.plot(k_distances_sorted, marker='o', linestyle='-', markersize=3)
plt.xlabel("Data Points (sorted)")
plt.ylabel(f"Distance to {min_samples}th Nearest Neighbor")
plt.title("K-Distance Plot for DBSCAN eps Selection")
plt.grid(True)
plt.show()

####################################################################################################

# Apply KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
patient_labels = kmeans.fit_predict(patient_embeddings)

####################################################################################################

### 3. Visualize the Embeddings

# Visualize the embeddings using techniques like **UMAP** or **t-SNE** to explore the clusters of patients.

####################################################################################################

import umap.umap_ as umap
import matplotlib.pyplot as plt

# Reduce the dimensionality for visualization.
reducer = umap.UMAP(random_state=42)
embeddings_umap = reducer.fit_transform(patient_embeddings)

# Visualize KMeans clusters
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
            c=patient_labels, cmap='viridis', s=10)
plt.title("UMAP Visualization of Patient Embeddings (KMeans)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label='KMeans Cluster')
plt.show()

# Visualize DBSCAN clusters
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1],
            c=patient_labels_dbscan, cmap='Spectral', s=10)
plt.title("UMAP Visualization of Patient Embeddings (DBSCAN)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label='DBSCAN Cluster')
plt.show()

####################################################################################################
