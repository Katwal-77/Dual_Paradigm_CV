"""
Complete Demo: Weather Classification vs Vehicle Detection
Shows both approaches side-by-side for final year project presentation
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import json

# Import both systems
from predict import WeatherPredictor
from vehicle_detector import VehicleDetector
import config
import vehicle_detection_config as vconfig


class CompleteDemoSystem:
    """Demonstrates both classification and detection"""
    
    def __init__(self):
        print("="*70)
        print("🎓 FINAL YEAR PROJECT: COMPUTER VISION DEMONSTRATION")
        print("="*70)
        print("\nInitializing both systems...\n")
        
        # Initialize weather classifier
        print("1️⃣  Loading Weather Classification Model...")
        try:
            model_path = config.MODELS_DIR / 'best_model.pth'
            if model_path.exists():
                self.weather_predictor = WeatherPredictor(str(model_path))
                print("   ✅ Weather classifier loaded successfully!")
            else:
                print("   ⚠️  No trained weather model found.")
                print("   💡 Run 'python train.py' first to train the model.")
                self.weather_predictor = None
        except Exception as e:
            print(f"   ❌ Error loading weather model: {e}")
            self.weather_predictor = None
        
        # Initialize vehicle detector
        print("\n2️⃣  Loading Vehicle Detection Model (YOLOv11)...")
        try:
            self.vehicle_detector = VehicleDetector()
            print("   ✅ Vehicle detector loaded successfully!")
        except Exception as e:
            print(f"   ❌ Error loading vehicle detector: {e}")
            self.vehicle_detector = None
        
        print("\n" + "="*70)
        print("✅ System Ready!")
        print("="*70 + "\n")
    
    def demo_classification(self, image_path):
        """
        Demonstrate image classification on weather images
        """
        print("\n" + "="*70)
        print("📸 DEMO 1: IMAGE CLASSIFICATION (Weather)")
        print("="*70)
        
        if not self.weather_predictor:
            print("❌ Weather classifier not available. Train the model first.")
            return
        
        print(f"\n🔍 Analyzing: {image_path}")
        
        # Get predictions
        predictions = self.weather_predictor.predict_image(image_path, top_k=5)
        
        print("\n📊 Classification Results:")
        print("-" * 50)
        for i, (class_name, confidence) in enumerate(predictions, 1):
            bar = "█" * int(confidence / 2)
            print(f"{i}. {class_name:12s} {confidence:5.1f}% {bar}")
        
        print("-" * 50)
        print(f"✅ Prediction: {predictions[0][0]} ({predictions[0][1]:.1f}% confidence)")
        
        # Visualize
        self.weather_predictor.visualize_prediction(image_path, save_path=None)
        
        return predictions
    
    def demo_detection(self, image_path):
        """
        Demonstrate object detection on vehicle images
        """
        print("\n" + "="*70)
        print("🚗 DEMO 2: OBJECT DETECTION (Vehicles)")
        print("="*70)
        
        if not self.vehicle_detector:
            print("❌ Vehicle detector not available.")
            return
        
        print(f"\n🔍 Detecting vehicles in: {image_path}")
        
        # Run detection
        output_path = vconfig.DETECTION_OUTPUT_DIR / f"demo_{Path(image_path).name}"
        results, vehicle_count = self.vehicle_detector.detect_image(
            image_path, 
            save_path=output_path,
            show=False
        )
        
        # Display results
        print("\n📊 Detection Results:")
        print("-" * 50)
        total = sum(vehicle_count.values())
        for vehicle_type, count in vehicle_count.items():
            if count > 0:
                bar = "█" * (count * 2)
                print(f"  {vehicle_type:12s}: {count:2d} {bar}")
        print("-" * 50)
        print(f"✅ Total vehicles detected: {total}")
        
        # Show image
        img = cv2.imread(str(output_path))
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title(f'Vehicle Detection Results: {total} vehicles found', fontsize=14)
            plt.tight_layout()
            plt.show()
        
        return vehicle_count
    
    def compare_approaches(self):
        """
        Create a comparison visualization showing both approaches
        """
        print("\n" + "="*70)
        print("📊 COMPARISON: Classification vs Detection")
        print("="*70)
        
        comparison = """
        
╔════════════════════════════════════════════════════════════════════╗
║                    IMAGE CLASSIFICATION                            ║
╠════════════════════════════════════════════════════════════════════╣
║  Input:  Entire image                                              ║
║  Output: Single label (e.g., "cloudy", "rainy")                    ║
║  Use:    Categorizing images into predefined classes               ║
║  Model:  ResNet50 (CNN)                                            ║
║  Task:   "What weather condition is shown?"                        ║
║                                                                     ║
║  Example:                                                           ║
║    Image → Model → "Cloudy" (85% confidence)                       ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════╗
║                      OBJECT DETECTION                              ║
╠════════════════════════════════════════════════════════════════════╣
║  Input:  Entire image                                              ║
║  Output: Multiple bounding boxes with labels                       ║
║  Use:    Finding and locating objects in images                    ║
║  Model:  YOLOv11                                                   ║
║  Task:   "What objects are present and where?"                     ║
║                                                                     ║
║  Example:                                                           ║
║    Image → Model → [Box1: "car" at (x1,y1,x2,y2), 92%]            ║
║                    [Box2: "truck" at (x3,y3,x4,y4), 87%]           ║
║                    [Box3: "bus" at (x5,y5,x6,y6), 95%]             ║
║                                                                     ║
╚════════════════════════════════════════════════════════════════════╝

        """
        print(comparison)
        
        # Create comparison table
        data = {
            'Aspect': [
                'Task Type',
                'Output',
                'Bounding Boxes',
                'Multiple Objects',
                'Localization',
                'Use Case (This Project)',
                'Model Used',
                'Accuracy Metric'
            ],
            'Classification': [
                'Categorization',
                'Single label',
                'No',
                'No',
                'No',
                'Weather recognition',
                'ResNet50',
                '76.19%'
            ],
            'Detection': [
                'Localization + Classification',
                'Multiple boxes + labels',
                'Yes',
                'Yes',
                'Yes',
                'Vehicle counting',
                'YOLOv11',
                'mAP (varies by dataset)'
            ]
        }
        
        print("\n📋 Detailed Comparison Table:")
        print("="*70)
        print(f"{'Aspect':<25} {'Classification':<22} {'Detection':<22}")
        print("="*70)
        for i in range(len(data['Aspect'])):
            print(f"{data['Aspect'][i]:<25} {data['Classification'][i]:<22} {data['Detection'][i]:<22}")
        print("="*70)
    
    def run_full_demo(self):
        """
        Run complete demonstration for presentation
        """
        print("\n" + "="*70)
        print("🎬 RUNNING FULL DEMONSTRATION")
        print("="*70)
        
        # Demo 1: Classification on weather images
        print("\n\n" + "🌤️  PART 1: WEATHER CLASSIFICATION" + "\n")
        weather_images = list(Path(config.DATA_DIR).glob("*/"))
        if weather_images:
            # Pick a random image from each class
            sample_classes = ['cloudy', 'rainy', 'sunny']
            for class_name in sample_classes:
                class_dir = config.DATA_DIR / class_name
                if class_dir.exists():
                    images = list(class_dir.glob("*.jpg"))
                    if images:
                        print(f"\n--- Testing on '{class_name}' image ---")
                        self.demo_classification(str(images[0]))
                        break
        
        # Demo 2: Detection on vehicle images
        print("\n\n" + "🚗 PART 2: VEHICLE DETECTION" + "\n")
        
        # Check if sample images exist
        sample_dir = vconfig.VEHICLE_DATA_DIR / "sample_images"
        if not sample_dir.exists() or not list(sample_dir.glob("*.jpg")):
            print("📥 No sample images found. Downloading...")
            from download_sample_images import download_sample_images
            sample_dir = download_sample_images()
        
        # Run detection on sample images
        vehicle_images = list(sample_dir.glob("*.jpg"))
        if vehicle_images:
            print(f"\n--- Testing on vehicle image ---")
            self.demo_detection(str(vehicle_images[0]))
        
        # Show comparison
        print("\n\n" + "📊 PART 3: COMPARISON" + "\n")
        self.compare_approaches()
        
        print("\n" + "="*70)
        print("✅ DEMONSTRATION COMPLETE!")
        print("="*70)
        print("\n💡 For your presentation:")
        print("   1. Show weather classification results (your trained model)")
        print("   2. Show vehicle detection results (YOLOv11)")
        print("   3. Explain the difference between the two approaches")
        print("   4. Discuss which is better for different use cases")
        print("="*70 + "\n")


def main():
    """Main demo function"""
    
    # Create demo system
    demo = CompleteDemoSystem()
    
    # Run full demonstration
    demo.run_full_demo()
    
    print("\n" + "="*70)
    print("🎓 READY FOR FINAL YEAR PROJECT PRESENTATION!")
    print("="*70)
    print("\nYou now have:")
    print("  ✅ Weather Classification (trained on your dataset)")
    print("  ✅ Vehicle Detection (YOLOv11)")
    print("  ✅ Comparison between both approaches")
    print("  ✅ Visual demonstrations")
    print("\nThis shows comprehensive understanding of Computer Vision!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

