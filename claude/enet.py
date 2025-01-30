import torch
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Dict
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class ImageFeatureAnalyzer:
    def __init__(self, 
                 model_name='efficientnet-b0', 
                 similarity_threshold=0.7,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Initialize EfficientNet model
        self.model = EfficientNet.from_pretrained(model_name)
        self.model = self.model.to(device)
        self.model.eval()
        
        # Initialize image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize graph
        self.G = nx.Graph()
        
    def get_image_paths_from_folder(self, folder_path: str) -> List[str]:
        folder_path = Path(folder_path)
        image_paths = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in self.image_extensions:
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        image_paths.append(str(file_path))
                    except Exception as e:
                        print(f"Skipping invalid image {file_path}: {str(e)}")
        
        return sorted(image_paths)

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            features = None
            def hook(module, input, output):
                nonlocal features
                features = output.detach().cpu().numpy()  # Fixed: Added detach()
            
            hook_handle = self.model._avg_pooling.register_forward_hook(hook)
            with torch.no_grad():  # Added to ensure no gradients are computed
                _ = self.model(image_tensor)
            hook_handle.remove()
            
            return features.reshape(features.shape[0], -1)
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_folder(self, folder_path: str, batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        image_paths = self.get_image_paths_from_folder(folder_path)
        if not image_paths:
            raise ValueError(f"No valid images found in {folder_path}")
        
        print(f"Processing {len(image_paths)} images...")
        
        all_features = []
        valid_paths = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            for path in batch_paths:
                features = self.extract_features(path)
                if features is not None:
                    all_features.append(features)
                    valid_paths.append(path)
            
            print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
        
        return np.vstack(all_features), valid_paths

    def create_similarity_graph(self, features: np.ndarray, image_paths: List[str]) -> nx.Graph:
        # Calculate pairwise similarities
        similarity_matrix = cosine_similarity(features)
        
        # Create nodes
        for idx, path in enumerate(image_paths):
            self.G.add_node(idx, image_path=Path(path).name)
        
        # Create edges for similarities above threshold
        num_images = len(image_paths)
        for i in range(num_images):
            for j in range(i + 1, num_images):
                similarity = similarity_matrix[i, j]
                if similarity > self.similarity_threshold:
                    self.G.add_edge(i, j, weight=similarity)
        
        return self.G

    def visualize_graph(self, 
                       node_size: int = 1000, 
                       with_labels: bool = True,
                       spring_k: float = 1,
                       save_path: str = None):
        plt.figure(figsize=(12, 8))
        
        pos = nx.spring_layout(self.G, k=spring_k)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.G, pos, 
                             node_color='lightblue',
                             node_size=node_size)
        
        # Draw edges
        edge_weights = [d['weight'] for (u, v, d) in self.G.edges(data=True)]
        nx.draw_networkx_edges(self.G, pos, 
                             width=[w * 2 for w in edge_weights],
                             alpha=0.5)
        
        # Add labels
        if with_labels:
            labels = nx.get_node_attributes(self.G, 'image_path')
            nx.draw_networkx_labels(self.G, pos, labels)
        
        plt.title("Image Feature Similarity Graph")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def get_graph_statistics(self) -> Dict:
        stats = {
            'num_nodes': self.G.number_of_nodes(),
            'num_edges': self.G.number_of_edges(),
            'average_degree': sum(dict(self.G.degree()).values()) / self.G.number_of_nodes(),
            'density': nx.density(self.G),
            'connected_components': nx.number_connected_components(self.G),
            'average_clustering': nx.average_clustering(self.G),
        }
        return stats

def main():
    # Example usage
    folder_path = "C:/Users/aydhi/OneDrive/Pictures/wall"
    
    # Initialize analyzer
    analyzer = ImageFeatureAnalyzer(
        similarity_threshold=0.7,
        model_name='efficientnet-b0'
    )
    
    # Process images and extract features
    features, image_paths = analyzer.process_folder(folder_path)
    print(f"\nSuccessfully extracted features from {len(image_paths)} images")
    print(f"Feature shape: {features.shape}")
    
    # Create similarity graph
    analyzer.create_similarity_graph(features, image_paths)
    
    # Print graph statistics
    stats = analyzer.get_graph_statistics()
    print("\nGraph Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Visualize the graph
    analyzer.visualize_graph(
        node_size=1000,
        with_labels=True,
        save_path="similarity_graph.png"
    )

if __name__ == "__main__":
    main()