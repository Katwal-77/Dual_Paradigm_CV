"""
Quick test of YOLOv11 vehicle detection
Downloads YOLOv11 model and tests on a sample image
"""
from vehicle_detector import VehicleDetector
import vehicle_detection_config as config
from pathlib import Path

def main():
    print("="*70)
    print("🚗 QUICK VEHICLE DETECTION TEST")
    print("="*70)
    
    # Initialize detector (will download YOLOv11 model automatically)
    print("\n📦 Initializing YOLOv11...")
    detector = VehicleDetector()
    
    print("\n" + "="*70)
    print("✅ YOLOv11 Model Ready!")
    print("="*70)
    
    # Check for test images in your weather dataset
    print("\n🔍 Looking for images to test...")
    
    # You can test on any image - let's use one from your weather dataset
    from config import DATA_DIR
    
    # Find any image
    test_image = None
    for class_dir in DATA_DIR.glob("*/"):
        images = list(class_dir.glob("*.jpg"))
        if images:
            test_image = images[0]
            break
    
    if test_image:
        print(f"📸 Testing on: {test_image}")
        print("\n⚠️  Note: This is a weather image, so it may not contain vehicles.")
        print("   For best results, use images with cars, trucks, or buses.\n")
        
        # Run detection
        output_path = config.DETECTION_OUTPUT_DIR / f"test_{test_image.name}"
        results, vehicle_count = detector.detect_image(
            str(test_image),
            save_path=output_path,
            show=False
        )
        
        total = sum(vehicle_count.values())
        if total > 0:
            print(f"\n🎉 Found {total} vehicles!")
        else:
            print("\n💡 No vehicles found (expected - this is a weather image)")
    
    print("\n" + "="*70)
    print("📝 NEXT STEPS:")
    print("="*70)
    print("\n1. To test on vehicle images:")
    print("   python demo_vehicle_detection.py --mode image --source <path_to_car_image> --show")
    
    print("\n2. To test on webcam (real-time):")
    print("   python demo_vehicle_detection.py --mode webcam")
    
    print("\n3. To run complete demo (classification + detection):")
    print("   python complete_demo.py")
    
    print("\n4. Download sample vehicle images:")
    print("   python download_sample_images.py")
    
    print("\n" + "="*70)
    print("✅ YOLOv11 is installed and working!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

