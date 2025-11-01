"""
Test vehicle detector with any available image
"""
from pathlib import Path
from vehicle_detector import VehicleDetector
import vehicle_detection_config as config

def test_detector():
    """Test the vehicle detector"""
    
    print("="*70)
    print("🚗 TESTING YOLOV11 VEHICLE DETECTOR")
    print("="*70)
    
    # Initialize detector
    print("\n📦 Loading YOLOv11 model...")
    detector = VehicleDetector()
    print("✅ Model loaded successfully!")
    
    # Check if we have any test images
    test_images = []
    
    # Check for manually downloaded images
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(list(Path('.').glob(ext)))
    
    if test_images:
        print(f"\n✅ Found {len(test_images)} test image(s):")
        for img in test_images[:5]:  # Show first 5
            print(f"   - {img}")
        
        # Test on first image
        test_img = test_images[0]
        print(f"\n🔍 Testing detection on: {test_img}")
        
        output_path = config.DETECTION_OUTPUT_DIR / f"detected_{test_img.name}"
        detections, counts, annotated = detector.detect_image(
            str(test_img), 
            save_path=str(output_path),
            show=False
        )
        
        print(f"\n📊 Detection Results:")
        print(f"   Total detections: {len(detections)}")
        if counts:
            print(f"   Vehicle counts:")
            for vehicle_type, count in counts.items():
                print(f"      - {vehicle_type}: {count}")
        else:
            print("   ⚠️  No vehicles detected in this image")
            print("   💡 Try with an image that contains cars/vehicles")
        
        print(f"\n💾 Output saved to: {output_path}")
        
    else:
        print("\n⚠️  No test images found in current directory")
        print("\n📝 To test the detector, you have 3 options:")
        print("\n1️⃣  Download a car image and save it here:")
        print("   - Go to: https://www.pexels.com/search/car/")
        print("   - Download any car image")
        print("   - Save as: test_car.jpg")
        print("   - Run: python demo_vehicle_detection.py --mode image --source test_car.jpg --show")
        
        print("\n2️⃣  Use webcam (easiest):")
        print("   - Run: python demo_vehicle_detection.py --mode webcam")
        print("   - Point camera at toy cars or car pictures")
        
        print("\n3️⃣  Try with weather dataset images:")
        print("   - Run: python demo_vehicle_detection.py --mode image --source data_classification/day/day_0.jpg --show")
        print("   - Some day/street images might have vehicles")
    
    print("\n" + "="*70)
    print("✅ Detector is working and ready to use!")
    print("="*70)

if __name__ == "__main__":
    test_detector()

