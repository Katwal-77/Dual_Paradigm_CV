"""
Create a single comprehensive results summary visualization
Perfect for presentations and reports
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
import config

# Set style
sns.set_style("white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def create_results_summary():
    """Create comprehensive results summary in one image"""
    
    # Load training history
    history_file = config.RESULTS_DIR / 'training_history.json'
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle('COMPUTER VISION PROJECT - COMPLETE RESULTS SUMMARY', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors
    color_train = '#2E86AB'
    color_val = '#A23B72'
    color_best = '#F18F01'
    
    # ============= ROW 1: Training Metrics =============
    
    # 1. Loss Curve
    ax1 = plt.subplot(3, 4, 1)
    ax1.plot(epochs, history['train_loss'], color=color_train, linewidth=2.5, label='Training')
    ax1.plot(epochs, history['val_loss'], color=color_val, linewidth=2.5, label='Validation')
    best_epoch = np.argmin(history['val_loss']) + 1
    ax1.axvline(x=best_epoch, color=color_best, linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#F8F9FA')
    
    # 2. Accuracy Curve
    ax2 = plt.subplot(3, 4, 2)
    ax2.plot(epochs, history['train_acc'], color=color_train, linewidth=2.5, label='Training')
    ax2.plot(epochs, history['val_acc'], color=color_val, linewidth=2.5, label='Validation')
    best_val_acc_epoch = np.argmax(history['val_acc']) + 1
    best_val_acc = max(history['val_acc'])
    ax2.axvline(x=best_val_acc_epoch, color=color_best, linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=best_val_acc, color=color_best, linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#F8F9FA')
    
    # 3. Learning Rate
    ax3 = plt.subplot(3, 4, 3)
    ax3.plot(epochs, history['learning_rates'], color='#06A77D', linewidth=2.5)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#F8F9FA')
    
    # 4. Overfitting Gap
    ax4 = plt.subplot(3, 4, 4)
    overfitting_gap = [train - val for train, val in zip(history['train_acc'], history['val_acc'])]
    ax4.fill_between(epochs, 0, overfitting_gap, alpha=0.6, color='#C73E1D')
    ax4.plot(epochs, overfitting_gap, color='#8B0000', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Gap (%)', fontweight='bold')
    ax4.set_title('Overfitting Gap (Train - Val)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#F8F9FA')
    
    # ============= ROW 2: Key Metrics =============
    
    # 5. Final Metrics Bar Chart
    ax5 = plt.subplot(3, 4, 5)
    metrics = ['Train\nAccuracy', 'Val\nAccuracy', 'Best Val\nAccuracy']
    values = [history['train_acc'][-1], history['val_acc'][-1], best_val_acc]
    colors_bars = [color_train, color_val, color_best]
    bars = ax5.bar(metrics, values, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=2)
    ax5.set_ylabel('Accuracy (%)', fontweight='bold')
    ax5.set_title('Final Accuracy Metrics', fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 105])
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_facecolor('#F8F9FA')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 6. Training Progress
    ax6 = plt.subplot(3, 4, 6)
    milestones = {
        'Start': (1, history['val_acc'][0]),
        'Epoch 10': (10, history['val_acc'][9]),
        'Epoch 20': (20, history['val_acc'][19]),
        'Best': (best_val_acc_epoch, best_val_acc),
        'Final': (len(epochs), history['val_acc'][-1])
    }
    
    milestone_epochs = [v[0] for v in milestones.values()]
    milestone_accs = [v[1] for v in milestones.values()]
    
    ax6.plot(epochs, history['val_acc'], color=color_val, linewidth=2, alpha=0.5)
    ax6.scatter(milestone_epochs, milestone_accs, color='red', s=150, zorder=5, edgecolor='black', linewidth=2)
    
    for i, (label, (ep, acc)) in enumerate(milestones.items()):
        ax6.annotate(f'{label}\n{acc:.1f}%', 
                    xy=(ep, acc), 
                    xytext=(10, 10 if i % 2 == 0 else -20),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax6.set_xlabel('Epoch', fontweight='bold')
    ax6.set_ylabel('Validation Accuracy (%)', fontweight='bold')
    ax6.set_title('Training Milestones', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.set_facecolor('#F8F9FA')
    
    # 7. Dataset Split
    ax7 = plt.subplot(3, 4, 7)
    total_images = 3360
    splits = ['Train\n(70%)', 'Validation\n(15%)', 'Test\n(15%)']
    split_counts = [int(total_images * 0.7), int(total_images * 0.15), int(total_images * 0.15)]
    colors_split = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax7.bar(splits, split_counts, color=colors_split, alpha=0.8, edgecolor='black', linewidth=2)
    ax7.set_ylabel('Number of Images', fontweight='bold')
    ax7.set_title('Dataset Split', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_facecolor('#F8F9FA')
    
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 30,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 8. Model Architecture Info
    ax8 = plt.subplot(3, 4, 8)
    ax8.axis('off')
    
    arch_text = """
    MODEL: ResNet50
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Architecture:
      • 50 Convolutional Layers
      • Residual Connections
      • Batch Normalization
      • Global Average Pooling
    
    Parameters:
      • Total: 23.5 Million
      • Trainable: 23.5M
    
    Pre-training:
      • ImageNet (1.2M images)
      • 1000 classes
      • Transfer Learning
    
    Output:
      • 16 Weather Classes
      • Softmax Activation
    """
    
    ax8.text(0.1, 0.5, arch_text, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))
    ax8.set_title('Model Architecture', fontsize=12, fontweight='bold')
    
    # ============= ROW 3: Summary Statistics =============
    
    # 9. Training Configuration
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    config_text = """
    TRAINING CONFIG
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Optimizer: Adam
    Initial LR: 0.001
    Batch Size: 32
    Max Epochs: 50
    
    Scheduler:
      ReduceLROnPlateau
      Factor: 0.1
      Patience: 5
    
    Early Stopping:
      Patience: 10 epochs
      Triggered: Yes (Epoch 27)
    
    Loss Function:
      Cross-Entropy Loss
    
    Regularization:
      Dropout: 0.5
      Weight Decay: 1e-4
    """
    
    ax9.text(0.1, 0.5, config_text, fontsize=9, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3, pad=1))
    ax9.set_title('Training Configuration', fontsize=12, fontweight='bold')
    
    # 10. Data Augmentation
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    
    aug_text = """
    DATA AUGMENTATION
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Applied to Training Set:
    
    ✓ Random Horizontal Flip
      Probability: 50%
    
    ✓ Random Rotation
      Range: ±15 degrees
    
    ✓ Color Jitter
      Brightness: ±20%
      Contrast: ±20%
    
    ✓ Random Affine
      Translation: ±10%
    
    ✓ Normalization
      Mean: [0.485, 0.456, 0.406]
      Std: [0.229, 0.224, 0.225]
    """
    
    ax10.text(0.1, 0.5, aug_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3, pad=1))
    ax10.set_title('Data Augmentation', fontsize=12, fontweight='bold')
    
    # 11. Key Results
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    results_text = f"""
    KEY RESULTS
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Best Performance:
      Epoch: {best_val_acc_epoch}
      Val Accuracy: {best_val_acc:.2f}%
    
    Final Performance:
      Train Acc: {history['train_acc'][-1]:.2f}%
      Val Acc: {history['val_acc'][-1]:.2f}%
      
    Training Duration:
      Total Epochs: {len(epochs)}
      Early Stop: Yes
      
    Overfitting:
      Gap: {overfitting_gap[-1]:.2f}%
      Status: Detected
      Mitigation: Early Stopping
    
    Model Saved:
      ✓ best_model.pth
      ✓ latest_model.pth
    """
    
    ax11.text(0.1, 0.5, results_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3, pad=1))
    ax11.set_title('Key Results', fontsize=12, fontweight='bold')
    
    # 12. Project Summary
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    summary_text = """
    PROJECT SUMMARY
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    CLASSIFICATION SYSTEM:
      Dataset: 3,360 images
      Classes: 16 weather types
      Model: ResNet50
      Accuracy: 76.19%
    
    DETECTION SYSTEM:
      Model: YOLOv11-nano
      Classes: 6 vehicle types
      Mode: Real-time
      Pre-trained: COCO
    
    ACHIEVEMENTS:
      ✓ Dual CV approach
      ✓ Transfer learning
      ✓ Early stopping
      ✓ LR scheduling
      ✓ Data augmentation
      ✓ Complete pipeline
    
    STATUS: Ready for Demo
    """
    
    ax12.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.5, pad=1))
    ax12.set_title('Project Summary', fontsize=12, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_dir = config.RESULTS_DIR / 'comprehensive_visualizations'
    output_dir.mkdir(exist_ok=True, parents=True)
    save_path = output_dir / 'COMPLETE_RESULTS_SUMMARY.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comprehensive results summary: {save_path}")
    plt.close()
    
    return save_path


if __name__ == "__main__":
    print("\n" + "="*80)
    print("📊 CREATING COMPREHENSIVE RESULTS SUMMARY")
    print("="*80 + "\n")
    
    path = create_results_summary()
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📁 Saved to: {path}")
    print("\nThis single image contains:")
    print("  • Training & validation curves")
    print("  • Accuracy metrics")
    print("  • Learning rate schedule")
    print("  • Overfitting analysis")
    print("  • Model architecture")
    print("  • Training configuration")
    print("  • Data augmentation details")
    print("  • Key results summary")
    print("\nPerfect for presentations and reports!")
    print("="*80 + "\n")

