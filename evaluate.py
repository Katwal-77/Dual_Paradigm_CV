"""
Model evaluation and visualization script
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import json

import config
from models import get_model
from data_loader import create_data_loaders
from utils import load_checkpoint


class ModelEvaluator:
    """Class for evaluating trained models"""
    
    def __init__(self, model, test_loader, device):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    def predict(self):
        """Get predictions for entire test set"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def evaluate(self):
        """Evaluate model and return metrics"""
        labels, preds, probs = self.predict()
        
        # Calculate metrics
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            labels, preds,
            target_names=config.CLASSES,
            output_dict=True
        )
        
        metrics = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'class_report': class_report
        }
        
        return metrics, labels, preds, probs
    
    def plot_confusion_matrix(self, labels, preds, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, preds)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=config.CLASSES,
            yticklabels=config.CLASSES,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix - Weather Classification', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_per_class_accuracy(self, class_report, save_path=None):
        """Plot per-class accuracy"""
        classes = config.CLASSES
        accuracies = [class_report[cls]['precision'] * 100 for cls in classes]
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(classes)), accuracies, color='steelblue', edgecolor='black')
        
        # Color code bars
        for i, bar in enumerate(bars):
            if accuracies[i] >= 90:
                bar.set_color('green')
            elif accuracies[i] >= 75:
                bar.set_color('orange')
            else:
                bar.set_color('red')
        
        plt.xlabel('Weather Class', fontsize=12, fontweight='bold')
        plt.ylabel('Precision (%)', fontsize=12, fontweight='bold')
        plt.title('Per-Class Precision - Weather Classification', fontsize=16, fontweight='bold')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Per-class accuracy plot saved to {save_path}")
        plt.close()
    
    def plot_training_history(self, history_path, save_path=None):
        """Plot training history"""
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Training history plot saved to {save_path}")
        plt.close()


def main():
    """Main evaluation function"""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60 + "\n")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}\n")
    
    # Load data
    print("Loading test data...")
    _, _, test_loader = create_data_loaders()
    
    # Load model
    print(f"Loading model: {config.MODEL_NAME}")
    model = get_model(
        model_name=config.MODEL_NAME,
        num_classes=config.NUM_CLASSES,
        pretrained=False
    )
    
    # Load best checkpoint
    checkpoint_path = config.MODELS_DIR / 'best_model.pth'
    if checkpoint_path.exists():
        model, _, epoch, val_acc = load_checkpoint(model, None, checkpoint_path)
        print(f"✅ Loaded checkpoint from epoch {epoch} (Val Acc: {val_acc:.2f}%)\n")
    else:
        print("⚠️  No checkpoint found! Using untrained model.\n")
    
    # Evaluate
    evaluator = ModelEvaluator(model, test_loader, device)
    metrics, labels, preds, probs = evaluator.evaluate()
    
    # Print results
    print("\n" + "="*60)
    print("📈 TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}%")
    print(f"Recall:    {metrics['recall']:.2f}%")
    print(f"F1-Score:  {metrics['f1_score']:.2f}%")
    print("="*60 + "\n")
    
    # Save metrics
    metrics_path = config.RESULTS_DIR / 'test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"📁 Metrics saved to {metrics_path}\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    evaluator.plot_confusion_matrix(
        labels, preds,
        save_path=config.RESULTS_DIR / 'confusion_matrix.png'
    )
    evaluator.plot_per_class_accuracy(
        metrics['class_report'],
        save_path=config.RESULTS_DIR / 'per_class_accuracy.png'
    )
    
    # Plot training history if available
    history_path = config.RESULTS_DIR / 'training_history.json'
    if history_path.exists():
        evaluator.plot_training_history(
            history_path,
            save_path=config.RESULTS_DIR / 'training_history.png'
        )
    
    print("\n✨ Evaluation completed successfully!")
    print(f"📁 All results saved to: {config.RESULTS_DIR}\n")


if __name__ == "__main__":
    main()

