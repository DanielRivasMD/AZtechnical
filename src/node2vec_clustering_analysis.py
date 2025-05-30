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

# Params for embedding
# Modify the ADJUST_PARAM values in embed_params based on your experimentation
embed_params = {
    "dim": ADJUST_PARAM,
    "num_walks": ADJUST_PARAM,
    "walk_length": ADJUST_PARAM,
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

### 2. Experiment with DBSCAN

# Please experiment with **DBSCAN** and compare its results with **KMeans**. Try to explore the differences between these two clustering algorithms.

####################################################################################################

# Apply KMeans clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
patient_labels = kmeans.fit_predict(patient_embeddings)

####################################################################################################

### 3. Visualize the Embeddings

# Visualize the embeddings using techniques like **UMAP** or **t-SNE** to explore the clusters of patients.

####################################################################################################
