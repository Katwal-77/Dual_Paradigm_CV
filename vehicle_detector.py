"""
YOLOv11 Vehicle Detection Script
Detects vehicles in images and videos
"""
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import json

import vehicle_detection_config as config


class VehicleDetector:
    """YOLOv11-based vehicle detector"""
    
    def __init__(self, model_path=None):
        """
        Initialize the vehicle detector
        
        Args:
            model_path: Path to custom model, or None to use pre-trained
        """
        print("🚗 Initializing YOLOv11 Vehicle Detector...")
        
        # Load model
        if model_path and Path(model_path).exists():
            print(f"📦 Loading custom model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"📦 Loading pre-trained YOLOv11 model: {config.YOLO_MODEL}")
            self.model = YOLO(config.YOLO_MODEL)
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  Using device: {self.device}")
        
        # Vehicle class IDs from COCO
        self.vehicle_classes = list(config.VEHICLE_CLASSES.keys())
        
        print("✅ Vehicle Detector initialized successfully!")
        print(f"   Detecting: {list(config.VEHICLE_CLASSES.values())}")
    
    def detect_image(self, image_path, save_path=None, show=False):
        """
        Detect vehicles in a single image
        
        Args:
            image_path: Path to input image
            save_path: Path to save output image (optional)
            show: Whether to display the result
            
        Returns:
            results: Detection results
            vehicle_count: Dictionary with count per vehicle type
        """
        print(f"\n🔍 Detecting vehicles in: {image_path}")
        
        # Run detection
        results = self.model.predict(
            source=image_path,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            classes=self.vehicle_classes,
            device=self.device,
            verbose=False
        )
        
        # Count vehicles by type
        vehicle_count = {name: 0 for name in config.VEHICLE_CLASSES.values()}
        total_vehicles = 0
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in config.VEHICLE_CLASSES:
                    vehicle_type = config.VEHICLE_CLASSES[cls_id]
                    vehicle_count[vehicle_type] += 1
                    total_vehicles += 1
        
        # Print results
        print(f"✅ Found {total_vehicles} vehicles:")
        for vehicle_type, count in vehicle_count.items():
            if count > 0:
                print(f"   - {vehicle_type}: {count}")
        
        # Save or show result
        if save_path or show:
            annotated_img = results[0].plot(
                line_width=config.BOX_THICKNESS,
                font_size=config.FONT_SIZE,
                labels=config.SHOW_LABELS,
                conf=config.SHOW_CONFIDENCE
            )
            
            if save_path:
                cv2.imwrite(str(save_path), annotated_img)
                print(f"💾 Saved result to: {save_path}")
            
            if show:
                cv2.imshow('Vehicle Detection', annotated_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        return results, vehicle_count
    
    def detect_video(self, video_path, save_path=None, show=False):
        """
        Detect vehicles in a video
        
        Args:
            video_path: Path to input video
            save_path: Path to save output video (optional)
            show: Whether to display the result
        """
        print(f"\n🎥 Detecting vehicles in video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer if saving
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = []
        
        # Process video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            results = self.model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                classes=self.vehicle_classes,
                device=self.device,
                verbose=False
            )
            
            # Count vehicles in this frame
            vehicle_count = 0
            for result in results:
                vehicle_count += len(result.boxes)
            
            total_detections.append(vehicle_count)
            
            # Annotate frame
            annotated_frame = results[0].plot(
                line_width=config.BOX_THICKNESS,
                font_size=config.FONT_SIZE
            )
            
            # Add frame info
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames} | Vehicles: {vehicle_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Save frame
            if writer:
                writer.write(annotated_frame)
            
            # Show frame
            if show:
                cv2.imshow('Vehicle Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            if frame_count % 30 == 0:
                print(f"   Processed {frame_count}/{total_frames} frames...")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Statistics
        avg_vehicles = np.mean(total_detections)
        max_vehicles = np.max(total_detections)
        
        print(f"\n✅ Video processing complete!")
        print(f"   Total frames: {frame_count}")
        print(f"   Average vehicles per frame: {avg_vehicles:.1f}")
        print(f"   Maximum vehicles in a frame: {max_vehicles}")
        
        if save_path:
            print(f"💾 Saved result to: {save_path}")
    
    def detect_webcam(self):
        """
        Real-time vehicle detection from webcam
        Press 'q' to quit
        """
        print("\n📷 Starting webcam detection...")
        print("   Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection every frame
            results = self.model.predict(
                source=frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                classes=self.vehicle_classes,
                device=self.device,
                verbose=False,
                stream=True
            )
            
            # Annotate frame
            for result in results:
                annotated_frame = result.plot(
                    line_width=config.BOX_THICKNESS,
                    font_size=config.FONT_SIZE
                )
                
                # Add info
                vehicle_count = len(result.boxes)
                cv2.putText(
                    annotated_frame,
                    f"Vehicles: {vehicle_count} | Press 'q' to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Show
                cv2.imshow('Vehicle Detection - Webcam', annotated_frame)
            
            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"✅ Webcam detection stopped. Processed {frame_count} frames.")
    
    def batch_detect(self, image_folder, output_folder=None):
        """
        Detect vehicles in multiple images
        
        Args:
            image_folder: Folder containing images
            output_folder: Folder to save results (optional)
        """
        image_folder = Path(image_folder)
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(image_folder.glob(f'*{ext}')))
            images.extend(list(image_folder.glob(f'*{ext.upper()}')))
        
        print(f"\n📁 Batch detection on {len(images)} images from {image_folder}")
        
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(exist_ok=True, parents=True)
        
        all_results = []
        
        for i, img_path in enumerate(images, 1):
            print(f"\n[{i}/{len(images)}] Processing: {img_path.name}")
            
            save_path = None
            if output_folder:
                save_path = output_folder / f"detected_{img_path.name}"
            
            results, vehicle_count = self.detect_image(img_path, save_path=save_path)
            
            all_results.append({
                'image': img_path.name,
                'vehicle_count': vehicle_count,
                'total': sum(vehicle_count.values())
            })
        
        # Save summary
        summary_path = config.VEHICLE_RESULTS_DIR / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        
        print(f"\n✅ Batch detection complete!")
        print(f"💾 Summary saved to: {summary_path}")
        
        return all_results


if __name__ == "__main__":
    # Example usage
    detector = VehicleDetector()
    
    print("\n" + "="*60)
    print("YOLOv11 Vehicle Detector Ready!")
    print("="*60)
    print("\nUsage examples:")
    print("1. Detect in image:")
    print("   detector.detect_image('path/to/image.jpg', save_path='output.jpg', show=True)")
    print("\n2. Detect in video:")
    print("   detector.detect_video('path/to/video.mp4', save_path='output.mp4')")
    print("\n3. Real-time webcam:")
    print("   detector.detect_webcam()")
    print("\n4. Batch detection:")
    print("   detector.batch_detect('path/to/images/', 'path/to/output/')")
    print("="*60)

