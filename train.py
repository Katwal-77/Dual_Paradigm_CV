"""
Training script for Weather Classification Model
"""
import os
import time
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import config
from models import get_model
from data_loader import create_data_loaders
from utils import save_checkpoint, load_checkpoint, EarlyStopping


class Trainer:
    """Trainer class for model training and validation"""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """Complete training loop"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting Training")
        print(f"{'='*60}")
        print(f"Model: {config.MODEL_NAME}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch Size: {config.BATCH_SIZE}")
        print(f"Learning Rate: {config.LEARNING_RATE}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"   Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                print(f"   ⭐ New best validation accuracy! Saving model...")
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_acc,
                    config.MODELS_DIR / 'best_model.pth'
                )
            
            # Save latest model
            save_checkpoint(
                self.model, self.optimizer, epoch, val_acc,
                config.MODELS_DIR / 'latest_model.pth'
            )
            
            # Early stopping
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training Complete!")
        print(f"{'='*60}")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = config.RESULTS_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"📁 Training history saved to {history_path}")


def main():
    """Main training function"""
    
    # Set random seed for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    print("\n" + "="*60)
    train_loader, val_loader, test_loader = create_data_loaders()
    
    # Create model
    print("="*60)
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=config.PRETRAINED
    )
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, device)
    history = trainer.train(config.NUM_EPOCHS)
    
    print("\n✨ Training pipeline completed successfully!")


if __name__ == "__main__":
    main()

