"""
Inference script for making predictions on new images
"""
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path

import config
from models import get_model
from utils import load_checkpoint


class WeatherPredictor:
    """Class for making predictions on new images"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load model
        self.model = get_model(
            model_name=config.MODEL_NAME,
            num_classes=config.NUM_CLASSES,
            pretrained=False
        )
        
        # Load checkpoint
        if Path(model_path).exists():
            self.model, _, _, _ = load_checkpoint(self.model, None, model_path)
            print(f"✅ Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path, top_k=3):
        """
        Predict weather class for a single image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with predictions and probabilities
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Format results
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': config.CLASSES[idx],
                'probability': float(prob * 100)
            })
        
        return {
            'image_path': str(image_path),
            'predictions': predictions,
            'top_prediction': predictions[0]
        }
    
    def visualize_prediction(self, image_path, save_path=None):
        """Visualize prediction with image and top predictions"""
        result = self.predict_image(image_path, top_k=5)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f"Predicted: {result['top_prediction']['class'].upper()}\n"
                     f"Confidence: {result['top_prediction']['probability']:.1f}%",
                     fontsize=14, fontweight='bold')
        
        # Display top predictions
        classes = [p['class'] for p in result['predictions']]
        probs = [p['probability'] for p in result['predictions']]
        
        colors = ['green' if i == 0 else 'steelblue' for i in range(len(classes))]
        bars = ax2.barh(classes, probs, color=colors, edgecolor='black')
        
        ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax2.text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return result


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(description='Weather Classification Prediction')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save visualization (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("\n" + "="*60)
    print("🔮 WEATHER PREDICTION")
    print("="*60 + "\n")
    
    # Create predictor
    predictor = WeatherPredictor(args.model, device=args.device)
    
    # Make prediction
    print(f"Analyzing image: {args.image}")
    result = predictor.visualize_prediction(args.image, save_path=args.output)
    
    # Print results
    print("\n" + "="*60)
    print("📊 PREDICTION RESULTS")
    print("="*60)
    print(f"\n🎯 Top Prediction: {result['top_prediction']['class'].upper()}")
    print(f"   Confidence: {result['top_prediction']['probability']:.2f}%\n")
    
    print("Top 5 Predictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"   {i}. {pred['class']:12s} - {pred['probability']:5.2f}%")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # For testing without command line args
    import sys
    if len(sys.argv) == 1:
        print("Usage: python predict.py --image <path_to_image>")
        print("\nExample:")
        print("  python predict.py --image data_classification/cloudy/cloudy_0.jpg")
        print("  python predict.py --image test_image.jpg --output prediction.png")
    else:
        main()

