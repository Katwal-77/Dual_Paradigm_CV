"""
Demo script for YOLOv11 Vehicle Detection
Run this to test the vehicle detector
"""
import argparse
from pathlib import Path
from vehicle_detector import VehicleDetector
import vehicle_detection_config as config


def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Vehicle Detection Demo')
    parser.add_argument('--mode', type=str, default='image', 
                       choices=['image', 'video', 'webcam', 'batch'],
                       help='Detection mode')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image/video/folder')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output')
    parser.add_argument('--show', action='store_true',
                       help='Display results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Update config
    config.CONFIDENCE_THRESHOLD = args.conf
    
    # Initialize detector
    print("="*70)
    print("🚗 YOLOv11 VEHICLE DETECTION SYSTEM")
    print("="*70)
    
    detector = VehicleDetector()
    
    # Run detection based on mode
    if args.mode == 'image':
        if not args.source:
            print("\n❌ Error: Please provide --source path to image")
            print("Example: python demo_vehicle_detection.py --mode image --source test.jpg --show")
            return
        
        output_path = args.output
        if not output_path:
            output_path = config.DETECTION_OUTPUT_DIR / f"detected_{Path(args.source).name}"
        
        detector.detect_image(args.source, save_path=output_path, show=args.show)
    
    elif args.mode == 'video':
        if not args.source:
            print("\n❌ Error: Please provide --source path to video")
            print("Example: python demo_vehicle_detection.py --mode video --source test.mp4")
            return
        
        output_path = args.output
        if not output_path:
            output_path = config.DETECTION_OUTPUT_DIR / f"detected_{Path(args.source).name}"
        
        detector.detect_video(args.source, save_path=output_path, show=args.show)
    
    elif args.mode == 'webcam':
        detector.detect_webcam()
    
    elif args.mode == 'batch':
        if not args.source:
            print("\n❌ Error: Please provide --source path to image folder")
            print("Example: python demo_vehicle_detection.py --mode batch --source ./images/")
            return
        
        output_folder = args.output
        if not output_folder:
            output_folder = config.DETECTION_OUTPUT_DIR / "batch_results"
        
        detector.batch_detect(args.source, output_folder=output_folder)
    
    print("\n" + "="*70)
    print("✅ Detection Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

