import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from model import get_model
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import torch.multiprocessing as mp

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_training=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.is_training = is_training
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all images and labels
        print(f"\nLoading images from {root_dir}")
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Directory {class_dir} does not exist!")
                continue
            
            image_files = list(class_dir.glob('*.jpeg'))
            print(f"Found {len(image_files)} images in {class_dir}")
            
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"Total images loaded: {len(self.images)}")
        if len(self.images) == 0:
            print("WARNING: No images found! Please check the dataset organization.")
            print("Expected structure:")
            print(f"{root_dir}/")
            print("├── NORMAL/")
            print("│   ├── image1.jpeg")
            print("│   └── image2.jpeg")
            print("└── PNEUMONIA/")
            print("    ├── image1.jpeg")
            print("    └── image2.jpeg")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply data augmentation for training
            if self.is_training:
                # Random horizontal flip
                if random.random() > 0.5:
                    image = transforms.functional.hflip(image)
                
                # Random rotation (-10 to 10 degrees)
                angle = random.uniform(-10, 10)
                image = transforms.functional.rotate(image, angle)
                
                # Random brightness and contrast
                brightness_factor = random.uniform(0.8, 1.2)
                contrast_factor = random.uniform(0.8, 1.2)
                image = transforms.functional.adjust_brightness(image, brightness_factor)
                image = transforms.functional.adjust_contrast(image, contrast_factor)
            
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    
    # Enable cudnn benchmarking for faster training
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix({'loss': running_loss/len(train_loader)})
        
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': val_loss/len(val_loader),
                    'acc': 100.*correct/total
                })
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * correct / total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}')
        print(f'Val Accuracy: {epoch_val_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save the best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'models/pneumonia_model.pth')
            print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")
    
    return train_losses, val_losses, val_accuracies

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms with smaller image size for faster processing
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # Reduced from 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = PneumoniaDataset('data/train', transform=transform, is_training=True)
    val_dataset = PneumoniaDataset('data/val', transform=transform, is_training=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("Error: Empty dataset! Please check the data organization and try again.")
        return
    
    # Calculate class weights for balanced training
    labels = train_dataset.labels
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    sample_weights = [class_weights[label].item() for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    # Create data loaders with increased batch size and num_workers
    print("\nCreating data loaders...")
    num_workers = min(4, mp.cpu_count())  # Use up to 4 workers
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64,  # Increased from 32
        sampler=sampler, 
        num_workers=num_workers,
        pin_memory=True  # Enable pin_memory for faster data transfer to GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=64,  # Increased from 32
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Initialize model, criterion, and optimizer
    print("\nInitializing model...")
    model = get_model(device, dropout_rate=0.3)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(  # Changed to AdamW for better regularization
        model.parameters(), 
        lr=0.001, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=15,  # Reduced from 20
        device=device
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    main() 