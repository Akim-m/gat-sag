import os
import pandas as pd
import numpy as np
import torch
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pickle
from tqdm import tqdm  # For progress display
import time
import gc
import faiss
from scipy.sparse import lil_matrix
from torchvision.models import ResNet50_Weights
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import matplotlib.pyplot as plt

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def visualize_graph(graph_file):
    """Visualize the graph stored in the specified file."""
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=50,
        node_color="blue",
        edge_color="gray",
        alpha=0.7,
        font_size=8
    )
    plt.title("Graph Visualization")
    plt.show()


def save_graph_as_png(graph_file, save_path):
    """Save the graph visualization as a PNG file."""
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Draw the graph
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=50,
        node_color="blue",
        edge_color="gray",
        alpha=0.7,
        font_size=8
    )
    plt.title("Graph Visualization")
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()


def log_time(function_name, start_time):
    """Log the execution time of a function."""
    elapsed_time = time.time() - start_time
    print(f"\n--- {function_name} completed in {elapsed_time:.2f} seconds ---")

class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            img_path = self.df.iloc[idx]['FilePath']
            label = self.df.iloc[idx]['Label']
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            return None, None

def extract_features_batch(model, dataloader, output_file, batch_size=32):
    """Extract features on the GPU in batches."""
    print("\n--- Starting extract_features_batch ---")
    start_time = time.time()

    model.eval()
    features_dict = {}
    
    with torch.no_grad():
        for batch_imgs, batch_labels in tqdm(dataloader, desc="Processing batches"):
            if batch_imgs is None:
                continue
            
            batch_imgs = batch_imgs.to(device)  # Move batch to GPU
            features = model(batch_imgs).cpu().numpy()  # Feature extraction on GPU
            for i, feature in enumerate(features):
                features_dict[len(features_dict)] = {'label': batch_labels[i], 'features': feature}
            
            torch.cuda.empty_cache()
        
        # Save extracted features
        with open(output_file, 'wb') as f:
            pickle.dump(features_dict, f)
    
    log_time("extract_features_batch", start_time)

def build_graph_from_features(feature_file, graph_file, similarity_threshold=0.5):
    """Build graph using GPU-accelerated similarity computation."""
    print("\n--- Starting build_graph_from_features ---")
    start_time = time.time()
    import networkx as nx
    
    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)
    
    G = nx.Graph()
    nodes = list(features_dict.keys())
    features = torch.tensor(
        np.array([features_dict[n]['features'] for n in nodes]),
        device=device
    )
    
    similarities = features @ features.T  # Compute similarity matrix
    norms = features.norm(dim=1).unsqueeze(0)
    similarities /= norms.T @ norms  # Normalize similarities
    
    for i in tqdm(range(len(nodes)), desc="Building graph"):
        for j in range(i + 1, len(nodes)):
            sim = similarities[i, j].item()  # Extract similarity
            if sim > similarity_threshold:
                print(f"Edge added between {nodes[i]} and {nodes[j]} with similarity {sim}")
                G.add_edge(nodes[i], nodes[j], weight=sim)

    
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    print(f"Graph saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    log_time("build_graph_from_features", start_time)

#using Faiss-cpu
def build_graph_from_features_faiss(feature_file, graph_file, similarity_threshold=0.5, top_k=10):
    """Build graph using FAISS for approximate nearest neighbors."""
    print("\n--- Starting build_graph_from_features (FAISS) ---")
    start_time = time.time()
    import networkx as nx
    
    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)
    
    G = nx.Graph()
    nodes = list(features_dict.keys())
    features = np.array([features_dict[n]['features'] for n in nodes])
    
    # Use FAISS for nearest neighbor search
    index = faiss.IndexFlatL2(features.shape[1])  # L2 distance
    index.add(features)  # Add features to FAISS index
    distances, indices = index.search(features, top_k)  # Find top_k nearest neighbors
    
    for i, (node, neighbors) in enumerate(zip(nodes, indices)):
        for j, dist in zip(neighbors, distances[i]):
            if i != j and dist < similarity_threshold:
                G.add_edge(node, nodes[j], weight=1 - dist)  # Convert distance to similarity
    
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    log_time("build_graph_from_features (FAISS)", start_time)

#Chunk based
def build_graph_from_features_chunked(feature_file, graph_file, similarity_threshold=0.5, chunk_size=64):
    """Build graph using chunk-based processing with progress bars."""
    print("\n--- Starting build_graph_from_features (Chunked) ---")
    start_time = time.time()
    import networkx as nx

    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)
    
    G = nx.Graph()
    nodes = list(features_dict.keys())
    features = torch.tensor(
        np.array([features_dict[n]['features'] for n in nodes]),
        device=device
    )

    outer_progress = tqdm(range(0, len(nodes), chunk_size), desc="Processing outer chunks")  # Outer loop
    for start_i in outer_progress:
        end_i = min(start_i + chunk_size, len(nodes))
        chunk_i = features[start_i:end_i]
        
        inner_progress = tqdm(range(start_i, len(nodes), chunk_size), desc="Processing inner chunks", leave=False)  # Inner loop
        for start_j in inner_progress:
            end_j = min(start_j + chunk_size, len(nodes))
            chunk_j = features[start_j:end_j]
            
            similarities = chunk_i @ chunk_j.T
            norms_i = chunk_i.norm(dim=1).unsqueeze(1)
            norms_j = chunk_j.norm(dim=1).unsqueeze(0)
            similarities = similarities / (norms_i @ norms_j)  # Normalize similarities
            
            for i, j in zip(*torch.where(similarities > similarity_threshold)):
                if start_i + i < start_j + j:  # Avoid duplicate edges
                    G.add_edge(
                        nodes[start_i + i], nodes[start_j + j],
                        weight=similarities[i, j].item()
                    )
            
            # Free memory
            del chunk_j, similarities
            torch.cuda.empty_cache()

        # Free memory
        del chunk_i
        torch.cuda.empty_cache()
    
    # Save graph
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    log_time("build_graph_from_features (Chunked)", start_time)

#sparse matrix

def build_graph_from_features_sparse(feature_file, graph_file, similarity_threshold=0.5):
    """Build graph using sparse similarity matrix."""
    print("\n--- Starting build_graph_from_features (Sparse Matrix) ---")
    start_time = time.time()
    import networkx as nx

    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)
    
    G = nx.Graph()
    nodes = list(features_dict.keys())
    features = torch.tensor(
        np.array([features_dict[n]['features'] for n in nodes]),
        device=device
    )
    
    n = len(nodes)
    sparse_matrix = lil_matrix((n, n), dtype=np.float32)
    
    for i in tqdm(range(n), desc="Building sparse matrix"):
        similarities = features[i] @ features.T
        norms = features.norm(dim=1)
        similarities = similarities / (norms[i] * norms)  # Normalize similarities
        for j in range(n):
            if i != j and similarities[j] > similarity_threshold:
                sparse_matrix[i, j] = similarities[j].item()
    
    # Build graph from sparse matrix
    G.add_weighted_edges_from(
        [(nodes[i], nodes[j], sparse_matrix[i, j]) for i, j in zip(*sparse_matrix.nonzero())]
    )
    
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    log_time("build_graph_from_features (Sparse Matrix)", start_time)


def compute_chunk_similarity(start_i, end_i, start_j, end_j, features, nodes, similarity_threshold):
    """Compute similarities between two chunks and return edges."""
    chunk_i = features[start_i:end_i]
    chunk_j = features[start_j:end_j]

    # Compute pairwise similarities
    similarities = np.dot(chunk_i, chunk_j.T)  # Matrix multiplication

    # Compute norms using NumPy
    norms_i = np.linalg.norm(chunk_i, axis=1).reshape(-1, 1)  # Column vector
    norms_j = np.linalg.norm(chunk_j, axis=1).reshape(1, -1)  # Row vector

    # Normalize similarities
    similarities = similarities / (norms_i @ norms_j)  # Element-wise division

    # Filter edges above the similarity threshold
    edges = []
    for i in range(similarities.shape[0]):
        for j in range(similarities.shape[1]):
            if similarities[i, j] > similarity_threshold:
                node_i = nodes[start_i + i]
                node_j = nodes[start_j + j]
                if start_i + i < start_j + j:  # Avoid duplicate edges
                    edges.append((node_i, node_j, similarities[i, j]))

    return edges


def process_task(args):
    """Wrapper to make compute_chunk_similarity picklable."""
    return compute_chunk_similarity(*args)


def build_graph_from_features_parallel(feature_file, graph_file, similarity_threshold=0.05, chunk_size=100, max_workers=4):
    """Build graph using parallelized chunk processing."""
    print("\n--- Starting build_graph_from_features_parallel ---")
    start_time = time.time()

    # Load features from file
    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)

    G = nx.Graph()
    nodes = list(features_dict.keys())
    # Convert features to NumPy (CPU-based) for multiprocessing
    features = np.array([features_dict[n]['features'] for n in nodes])

    # Prepare tasks for parallel processing
    tasks = []
    for start_i in range(0, len(nodes), chunk_size):
        end_i = min(start_i + chunk_size, len(nodes))
        for start_j in range(start_i, len(nodes), chunk_size):
            end_j = min(start_j + chunk_size, len(nodes))
            tasks.append((start_i, end_i, start_j, end_j, features, nodes, similarity_threshold))

    # Process tasks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="Processing chunks"))

    # Combine results into the graph
    edge_count = 0
    for edges in results:
        if edges:
            G.add_weighted_edges_from(edges)
            edge_count += len(edges)
    
    print(f"Total edges added: {edge_count}")
    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save the graph
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)

    log_time("build_graph_from_features_parallel", start_time)

from concurrent.futures import ThreadPoolExecutor

def build_graph_from_features_parallel_gpu(feature_file, graph_file, similarity_threshold=0.05, chunk_size=100, max_workers=4):
    """Build graph using parallelized chunk processing with GPU support."""
    print("\n--- Starting build_graph_from_features_parallel (GPU-enabled) ---")
    start_time = time.time()

    # Load features from file
    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)

    G = nx.Graph()
    nodes = list(features_dict.keys())
    features = torch.tensor(
        np.array([features_dict[n]['features'] for n in nodes]),
        device=device
    )  # Keep features on GPU

    # Prepare tasks for parallel processing
    tasks = []
    for start_i in range(0, len(nodes), chunk_size):
        end_i = min(start_i + chunk_size, len(nodes))
        for start_j in range(start_i, len(nodes), chunk_size):
            end_j = min(start_j + chunk_size, len(nodes))
            tasks.append((start_i, end_i, start_j, end_j, features, nodes, similarity_threshold))

    # Process tasks in parallel using threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_task, tasks), total=len(tasks), desc="Processing chunks"))

    # Combine results into the graph
    edge_count = 0
    for edges in results:
        if edges:
            G.add_weighted_edges_from(edges)
            edge_count += len(edges)

    print(f"Total edges added: {edge_count}")
    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save the graph
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)

    log_time("build_graph_from_features_parallel (GPU-enabled)", start_time)

def build_graph_from_features_chunked_gpu(feature_file, graph_file, similarity_threshold=0.05, chunk_size=100):
    """Build graph using chunk-based processing entirely on the GPU, with progress for inner chunks."""
    print("\n--- Starting build_graph_from_features_chunked (GPU-enabled) ---")
    start_time = time.time()

    # Load features from file
    with open(feature_file, 'rb') as f:
        features_dict = pickle.load(f)

    G = nx.Graph()
    nodes = list(features_dict.keys())
    features = torch.tensor(
        np.array([features_dict[n]['features'] for n in nodes]),
        device=device
    )  # Keep features on GPU

    # Process chunks
    outer_progress = tqdm(range(0, len(nodes), chunk_size), desc="Processing outer chunks")
    for start_i in outer_progress:
        end_i = min(start_i + chunk_size, len(nodes))
        chunk_i = features[start_i:end_i]

        inner_progress = tqdm(range(start_i, len(nodes), chunk_size), desc="Processing inner chunks", leave=False)
        for start_j in inner_progress:
            end_j = min(start_j + chunk_size, len(nodes))
            chunk_j = features[start_j:end_j]

            # Compute similarities on GPU
            similarities = chunk_i @ chunk_j.T
            norms_i = chunk_i.norm(dim=1).unsqueeze(1)
            norms_j = chunk_j.norm(dim=1).unsqueeze(0)
            similarities = similarities / (norms_i @ norms_j)

            # Add edges above the similarity threshold
            for i, j in zip(*torch.where(similarities > similarity_threshold)):
                if start_i + i < start_j + j:  # Avoid duplicate edges
                    G.add_edge(
                        nodes[start_i + i.item()], nodes[start_j + j.item()],
                        weight=similarities[i, j].item()
                    )

            torch.cuda.empty_cache()  # Free GPU memory after each inner chunk

        torch.cuda.empty_cache()  # Free GPU memory after each outer chunk

    print(f"Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Save the graph
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)

    log_time("build_graph_from_features_chunked (GPU-enabled)", start_time)


def main(base_folder, batch_size=64):
    """Main pipeline for feature extraction and graph building."""
    print("\n--- Starting main ---")
    start_time = time.time()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Scan dataset
    print("Scanning dataset...")
    data = [(os.path.join(root, file), os.path.basename(root)) for root, _, files in os.walk(base_folder) for file in files]
    df = pd.DataFrame(data, columns=['FilePath', 'Label'])
    
    dataset = ImageDataset(df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load pretrained model on GPU
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    
    feature_file = "features.pkl"
    # extract_features_batch(model, dataloader, feature_file, batch_size=batch_size)
    
    # Build graph using extracted features
    graph_file = "graph.pkl"
    build_graph_from_features_chunked_gpu(feature_file, graph_file)
    
    # log_time("main", start_time)
    save_graph_as_png("graph.pkl", "graph_visualization.png")



if __name__ == "__main__":
    main("lung_image_sets", batch_size=16)
