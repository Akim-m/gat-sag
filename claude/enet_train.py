import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class EfficientNetTrainer:
    def __init__(self, data_folder, num_classes=2, model_name='efficientnet_b0', lr=0.001, batch_size=32, epochs=10, device=None):
        self.data_folder = data_folder
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)
        self.model = self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)

        # Data transforms
        self.transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # Prepare data
        self.class_names, self.train_loader, self.val_loader = self._prepare_data()

    def _prepare_data(self):
        image_paths, labels, class_names = [], [], []
        for idx, class_name in enumerate(sorted(os.listdir(self.data_folder))):
            class_folder = os.path.join(self.data_folder, class_name)
            if not os.path.isdir(class_folder):
                continue
            class_names.append(class_name)
            for img_name in os.listdir(class_folder):
                if img_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_paths.append(os.path.join(class_folder, img_name))
                    labels.append(idx)

        X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)
        train_dataset = CustomImageDataset(X_train, y_train, self.transforms['train'])
        val_dataset = CustomImageDataset(X_val, y_val, self.transforms['val'])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        return class_names, train_loader, val_loader

    def _train_one_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(self.train_loader, desc='Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return running_loss / len(self.train_loader), 100 * correct / total

    def _validate(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc='Validating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        return running_loss / len(self.val_loader), 100 * correct / total

    def train(self):
        best_val_loss = float('inf')
        for epoch in range(self.epochs):
            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._validate()
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                print('Best model saved!')

    def save_model(self, path='final_model.pth'):
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

if __name__ == "__main__":
    data_folder = "lung_image_sets"  # Folder should contain class subfolders
    trainer = EfficientNetTrainer(data_folder, num_classes=2, model_name='efficientnet_b0', epochs=10)
    print(f"Classes found: {trainer.class_names}")
    trainer.train()
    trainer.save_model()
