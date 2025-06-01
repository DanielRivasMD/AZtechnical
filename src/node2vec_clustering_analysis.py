#!/usr/bin/env python
import os
import sys
import logging

####################################################################################################
# Setup colored logging
####################################################################################################

# ANSI color codes for each logging level
COLORS = {
    'DEBUG': '\033[0;37m',    # White
    'INFO': '\033[0;32m',     # Green
    'WARNING': '\033[0;33m',  # Yellow
    'ERROR': '\033[0;31m',    # Red
    'CRITICAL': '\033[1;31m'  # Bold Red
}
RESET = "\033[0m"

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{RESET}"
        return super().format(record)

# Create logger and set level
logger = logging.getLogger("analysis_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)

logger.info("Starting clustering analysis script with colored logging")

####################################################################################################
# Create graph directory where plots will be saved
####################################################################################################

GRAPH_DIR = 'graph'
if not os.path.exists(GRAPH_DIR):
    os.makedirs(GRAPH_DIR)
    logger.info("Created graph directory: %s", GRAPH_DIR)
else:
    logger.info("Graph directory already exists: %s", GRAPH_DIR)

####################################################################################################
# Dataframes
####################################################################################################

import pandas as pd
logger.info("pandas imported successfully")

####################################################################################################
# Numerical
####################################################################################################

import numpy as np
logger.info("numpy imported successfully")

####################################################################################################
# node2vec
####################################################################################################

from pecanpy.graph import AdjlstGraph
from pecanpy import pecanpy as node2vec
logger.info("pecanpy imported successfully")

####################################################################################################
# Graphics
####################################################################################################

import matplotlib.pyplot as plt
logger.info("matplotlib imported successfully")

####################################################################################################
# Algorithms
####################################################################################################

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
import umap.umap_ as umap
logger.info("sklearn and UMAP imported successfully")

####################################################################################################
# Load the data
####################################################################################################

# Bind the data path to a variable
DATA_PATH = 'data/2025_MLE_take_home_task/omics_embedding_take_home_2025.parquet'
logger.info("Loading data from '%s'", DATA_PATH)
df = pd.read_parquet(DATA_PATH)
logger.info("Data loaded successfully\nPreview:\n%s", df.head().to_string())

####################################################################################################
# Extract unique node IDs and edges
####################################################################################################

logger.info("Extracting unique node IDs and edges")
node_ids = pd.unique(df[['source_id', 'target_id']].values.ravel())
edges_iter = df[['source_id', 'target_id']].itertuples(index=False, name=None)

####################################################################################################
# Initialize graph
####################################################################################################

logger.info("Initializing graph")
graph = AdjlstGraph()

####################################################################################################
# Add nodes
####################################################################################################

logger.info("Adding nodes to graph")
for node_id in node_ids:
    graph.add_node(node_id)

####################################################################################################
# Add edges
####################################################################################################

logger.info("Adding edges to graph")
for source, target in edges_iter:
    graph.add_edge(source, target, directed=False)

####################################################################################################
# Select the Right Parameters for Node2Vec
####################################################################################################
logger.info("Configuring Node2Vec parameters")

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
embed_params = {
    "dim": adjust_dim,
    "num_walks": adjust_num_walks,
    "walk_length": adjust_walk_length,
    "window_size": 5,
    "epochs": 5,
    "verbose": True,
}

####################################################################################################
# Initialize and Train Node2Vec Model
####################################################################################################

logger.info("Initializing Node2Vec model")
model = node2vec.PreComp(init_params)

logger.info("Loading graph into Node2Vec model")
model = model.from_adjlst_graph(adjlst_graph=graph)

logger.info("Training Node2Vec embedding")
node_embeddings = model.embed(**embed_params)
logger.info("Embedding training complete")

####################################################################################################
# Filter patient embeddings
####################################################################################################

logger.info("Extracting patient IDs from data")
patients = set(df[df["source_type"] == "Patient"]["source_id"]) | \
           set(df[df["target_type"] == "Patient"]["target_id"])

logger.info("Filtering embeddings for patient nodes")
node_ids_array = np.array(node_ids)
patient_mask = np.isin(node_ids_array, list(patients))
patient_embeddings = node_embeddings[patient_mask]
patient_ids = node_ids_array[patient_mask]
logger.info("Patient embeddings ready: %d patients", len(patient_ids))

####################################################################################################
# PCA Visualization
####################################################################################################

logger.info("Performing PCA visualization")
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(patient_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=10, color='blue', alpha=0.7)
plt.title("PCA Projection of Patient Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
pca_filepath = os.path.join(GRAPH_DIR, "pca_projection.png")
plt.savefig(pca_filepath)
plt.close()
logger.info("PCA projection saved to %s", pca_filepath)

####################################################################################################
# t-SNE Visualization
####################################################################################################

logger.info("Performing t-SNE visualization")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(patient_embeddings)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=10, color='red', alpha=0.7)
plt.title("t-SNE Projection of Patient Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.grid(True)
tsne_filepath = os.path.join(GRAPH_DIR, "tsne_projection.png")
plt.savefig(tsne_filepath)
plt.close()
logger.info("t-SNE projection saved to %s", tsne_filepath)

####################################################################################################
# Experiment with DBSCAN
####################################################################################################

logger.info("Applying DBSCAN clustering")
dbscan = DBSCAN(eps=0.5, min_samples=5)
patient_labels_dbscan = dbscan.fit_predict(patient_embeddings)
logger.info("DBSCAN clustering complete")

####################################################################################################
# Nearest Neighbors for DBSCAN eps selection
####################################################################################################

logger.info("Computing Nearest Neighbors for DBSCAN")
min_samples = 5
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors_fit = neighbors.fit(patient_embeddings)
distances, indices = neighbors_fit.kneighbors(patient_embeddings)
k_distances = distances[:, min_samples - 1]

k_distances_sorted = np.sort(k_distances)
logger.info("Plotting k-distance graph for DBSCAN eps selection")
plt.figure(figsize=(8, 6))
plt.plot(k_distances_sorted, marker='o', linestyle='-', markersize=3)
plt.xlabel("Data Points (sorted)")
plt.ylabel(f"Distance to {min_samples}th Nearest Neighbor")
plt.title("K-Distance Plot for DBSCAN eps Selection")
plt.grid(True)
kdist_filepath = os.path.join(GRAPH_DIR, "k_distance_plot.png")
plt.savefig(kdist_filepath)
plt.close()
logger.info("K-distance plot saved to %s", kdist_filepath)

####################################################################################################
# Experiment with KMeans clustering and evaluate
####################################################################################################

logger.info("Applying KMeans clustering")
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
patient_labels = kmeans.fit_predict(patient_embeddings)
logger.info("KMeans clustering complete with %d clusters", n_clusters)

logger.info("Evaluating clustering for different values of k")
range_n_clusters = list(range(2, 11))
inertias = []
silhouette_scores = []
for k in range_n_clusters:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans_temp.fit_predict(patient_embeddings)
    inertias.append(kmeans_temp.inertia_)
    score = silhouette_score(patient_embeddings, labels)
    silhouette_scores.append(score)
    logger.info("For k = %d: inertia = %.2f, silhouette score = %.4f", k, kmeans_temp.inertia_, score)

logger.info("Plotting evaluation metrics (Elbow Method & Silhouette Analysis)")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range_n_clusters, inertias, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range_n_clusters, silhouette_scores, marker='o', color='red')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis for Optimal k")
plt.grid(True)

plt.tight_layout()
eval_filepath = os.path.join(GRAPH_DIR, "cluster_evaluation.png")
plt.savefig(eval_filepath)
plt.close()
logger.info("Cluster evaluation plot saved to %s", eval_filepath)

####################################################################################################
# Visualize the Embeddings using UMAP
####################################################################################################

logger.info("Performing UMAP dimensionality reduction")
reducer = umap.UMAP(random_state=42)
embeddings_umap = reducer.fit_transform(patient_embeddings)

####################################################################################################
# UMAP Visualization: KMeans clusters
####################################################################################################

logger.info("Visualizing UMAP projection of KMeans clusters")
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=patient_labels, cmap='viridis', s=10)
plt.title("UMAP Visualization of Patient Embeddings (KMeans)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label='KMeans Cluster')
umap_kmeans_filepath = os.path.join(GRAPH_DIR, "umap_kmeans.png")
plt.savefig(umap_kmeans_filepath)
plt.close()
logger.info("UMAP KMeans plot saved to %s", umap_kmeans_filepath)

####################################################################################################
# UMAP Visualization: DBSCAN clusters
####################################################################################################

logger.info("Visualizing UMAP projection of DBSCAN clusters")
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], c=patient_labels_dbscan, cmap='Spectral', s=10)
plt.title("UMAP Visualization of Patient Embeddings (DBSCAN)")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
plt.colorbar(label='DBSCAN Cluster')
umap_dbscan_filepath = os.path.join(GRAPH_DIR, "umap_dbscan.png")
plt.savefig(umap_dbscan_filepath)
plt.close()
logger.info("UMAP DBSCAN plot saved to %s", umap_dbscan_filepath)

####################################################################################################
logger.info("Clustering analysis completed successfully")
####################################################################################################
