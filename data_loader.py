"""
Data loading and preprocessing module
"""
import os
import random
from pathlib import Path
from typing import Tuple, List
import shutil

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import config


class WeatherDataset(Dataset):
    """Custom Dataset for Weather Classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment=True):
    """Get image transformations for training and validation"""
    
    if augment:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def prepare_data():
    """Prepare and split dataset into train, validation, and test sets"""
    
    print("📂 Loading dataset...")
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(config.CLASSES):
        class_dir = config.DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_name} directory not found!")
            continue
        
        class_images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        image_paths.extend([str(img) for img in class_images])
        labels.extend([class_idx] * len(class_images))
    
    print(f"✅ Found {len(image_paths)} images across {config.NUM_CLASSES} classes")
    
    # Split data: train, val, test
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, 
        test_size=(config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=labels
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.RANDOM_SEED,
        stratify=y_temp
    )
    
    print(f"📊 Data split:")
    print(f"   Training:   {len(X_train)} images ({len(X_train)/len(image_paths)*100:.1f}%)")
    print(f"   Validation: {len(X_val)} images ({len(X_val)/len(image_paths)*100:.1f}%)")
    print(f"   Testing:    {len(X_test)} images ({len(X_test)/len(image_paths)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_data_loaders(batch_size=None):
    """Create PyTorch DataLoaders for training, validation, and testing"""
    
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    
    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    
    # Get transforms
    train_transform, val_transform = get_transforms(augment=True)
    
    # Create datasets
    train_dataset = WeatherDataset(X_train, y_train, transform=train_transform)
    val_dataset = WeatherDataset(X_val, y_val, transform=val_transform)
    test_dataset = WeatherDataset(X_test, y_test, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print("✅ Data loaders created successfully!\n")
    
    return train_loader, val_loader, test_loader


def get_class_distribution(labels):
    """Get distribution of classes in the dataset"""
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip([config.CLASSES[i] for i in unique], counts))
    return distribution


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")

