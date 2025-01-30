import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import pandas as pd

def train_resnet(data_dir="lung_image_sets/", batch_size=32, num_epochs=10, lr=0.001, model_save_path="resnet_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    
    # Load ResNet50 without pre-trained weights
    model.load_state_dict(torch.load("resnet_model.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify final layer for classification
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def extract_features(data_dir="lung_image_sets/", model_path="resnet_model.pth", output_csv="features.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    num_classes = len(dataset.classes)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Ensure consistency with training
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Remove classification layer for feature extraction
    model.fc = nn.Identity()
    model = model.to(device)
    model.eval()
    
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            features = model(images).cpu().numpy().flatten()
            features_list.append(features)
            labels_list.append(labels.item())
    
    df = pd.DataFrame(features_list)
    df["label"] = labels_list
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

if __name__ == "__main__":
    # train_resnet()
    extract_features()
