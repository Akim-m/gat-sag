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

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
            if similarities[i, j] > similarity_threshold:
                G.add_edge(nodes[i], nodes[j], weight=similarities[i, j].item())
    
    with open(graph_file, 'wb') as f:
        pickle.dump(G, f)
    
    log_time("build_graph_from_features", start_time)

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
    model = resnet50(pretrained=True).to(device)
    
    feature_file = "features.pkl"
    # extract_features_batch(model, dataloader, feature_file, batch_size=batch_size)
    
    # Build graph using extracted features
    graph_file = "graph.pkl"
    build_graph_from_features(feature_file, graph_file)
    
    log_time("main", start_time)

if __name__ == "__main__":
    main("lung_image_sets", batch_size=16)
