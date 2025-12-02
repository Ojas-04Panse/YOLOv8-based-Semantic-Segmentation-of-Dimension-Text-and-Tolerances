"""
Deep Learning Pipeline for Technical Drawing Analysis

Handles TWO separate datasets:
1. View Detection Dataset (shapes, views, lines)
2. Dimension Detection Dataset (separated dimension, tolerance, etc. components)

Dataset Structure:
dataset_views/
  images/
    - drawing1.jpg
  labels/
    - drawing1.txt  (class: 0=view, 1=line, 2=circle, 3=rectangle)

dataset_dimensions/
  images/
    - dim001.jpg
  labels/
    - dim001.txt  (NEW CLASSES: 0=dimension, 1=tolerance_upper, 2=tolerance_lower, 3=tolerance)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Set the backend to 'Agg' (a non-interactive backend for file generation)
matplotlib.use('Agg')
from pathlib import Path
import yaml
import shutil
from typing import List, Dict, Tuple
import torch

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLOv8 not installed. Run: pip install ultralytics")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("EasyOCR not installed. Run: pip install easyocr")


class DualDatasetManager:
    """Manages both view and dimension datasets"""
    
    def __init__(self, view_dataset_path: str, dimension_dataset_path: str):
        self.view_path = Path(view_dataset_path)
        self.dim_path = Path(dimension_dataset_path)
        
        # --- PATCH 1: Updated Dimension Class Definitions ---
        self.view_classes = ['view', 'line', 'circle', 'rectangle', 'arc']
        # ONLY includes the four new specific text classes (assuming non-text lines are filtered out or handled separately)
        self.dim_classes = ['dimension', 'tolerance_upper', 'tolerance_lower', 'tolerance'] 
        # --- END PATCH 1 ---
        
    def prepare_both_datasets(self, train_ratio=0.7, val_ratio=0.2):
        """Prepare both datasets with proper splits"""
        print("=" * 60)
        print("Preparing BOTH Datasets")
        print("=" * 60)
        
        # Prepare view dataset
        print("\n[1] Preparing VIEW Detection Dataset...")
        view_yaml = self._prepare_single_dataset(
            self.view_path, 
            self.view_classes,
            'view_dataset.yaml',
            train_ratio,
            val_ratio
        )
        
        # Prepare dimension dataset
        print("\n[2] Preparing DIMENSION Detection Dataset...")
        dim_yaml = self._prepare_single_dataset(
            self.dim_path,
            self.dim_classes,
            'dimension_dataset.yaml',
            train_ratio,
            val_ratio
        )
        
        print("\n✓ Both datasets prepared successfully!")
        return view_yaml, dim_yaml
    
    def _prepare_single_dataset(self, dataset_path: Path, class_names: List[str], 
                               yaml_name: str, train_ratio: float, val_ratio: float):
        """Prepare a single dataset"""
        # Create structure
        splits = ['train', 'val', 'test']
        for split in splits:
            (dataset_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Get images from root images folder
        images_path = dataset_path / 'images'
        labels_path = dataset_path / 'labels'
        
        if not images_path.exists():
            print(f"  Warning: {images_path} not found. Skipping...")
            return None
        
        image_files = list(images_path.glob('*.[jp][pn]g')) + list(images_path.glob('*.jpeg'))
        
        if len(image_files) == 0:
            print(f"  Warning: No images found in {images_path}")
            return None
        
        print(f"  Found {len(image_files)} images")
        
        # Shuffle and split
        np.random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits_dict = {
            'train': image_files[:n_train],
            'val': image_files[n_train:n_train+n_val],
            'test': image_files[n_train+n_val:]
        }
        
        # Copy files to splits
        for split_name, files in splits_dict.items():
            for img_file in files:
                # Copy image
                dst_img = dataset_path / split_name / 'images' / img_file.name
                shutil.copy(img_file, dst_img)
                
                # Copy corresponding label
                label_file = labels_path / (img_file.stem + '.txt')
                if label_file.exists():
                    dst_label = dataset_path / split_name / 'labels' / label_file.name
                    shutil.copy(label_file, dst_label)
        
        print(f"  Split: {n_train} train, {n_val} val, {n_total-n_train-n_val} test")
        
        # Create YAML config
        config = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = str(dataset_path / yaml_name)
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
        
        print(f"  Config saved: {yaml_path}")
        return yaml_path
    
    def visualize_both_datasets(self, num_samples: int = 3):
        """Visualize samples from both datasets side by side"""
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 10))
        
        # Visualize view dataset
        self._visualize_dataset(self.view_path, axes[0], "VIEW Dataset", num_samples)
        
        # Visualize dimension dataset
        self._visualize_dataset(self.dim_path, axes[1], "DIMENSION Dataset", num_samples)
        
        plt.tight_layout()
        plt.savefig('both_datasets_visualization.png', dpi=150, bbox_inches='tight')
        print("\n✓ Visualization saved: both_datasets_visualization.png")
    
    def _visualize_dataset(self, dataset_path: Path, axes, title: str, num_samples: int):
        """Visualize samples from a dataset"""
        image_files = list((dataset_path / 'train' / 'images').glob('*.[jp][pn]g'))[:num_samples]
        
        for idx, img_file in enumerate(image_files):
            if idx >= len(axes):
                break
                
            img = cv2.imread(str(img_file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            
            # Read labels
            label_file = dataset_path / 'train' / 'labels' / (img_file.stem + '.txt')
            if label_file.exists():
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id, x_c, y_c, width, height = map(float, parts[:5])
                            
                            # Convert to pixel coords
                            x_c, y_c, width, height = x_c * w, y_c * h, width * w, height * h
                            x1 = int(x_c - width/2)
                            y1 = int(y_c - height/2)
                            x2 = int(x_c + width/2)
                            y2 = int(y_c + height/2)
                            
                            # Color based on class
                            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                            color = colors[int(cls_id) % len(colors)]
                            
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, f"C{int(cls_id)}", (x1, y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{title}\n{img_file.name}")
            axes[idx].axis('off')
        
        # Fill empty subplots
        for idx in range(len(image_files), len(axes)):
            axes[idx].axis('off')


class TwoStageDetector:
    """Two-stage detector: First detects views, then detects dimensions"""
    
    def __init__(self, view_model_path: str = None, dim_model_path: str = None):
        if not YOLO_AVAILABLE:
            raise ImportError("Install ultralytics: pip install ultralytics")
        
        # Initialize models
        self.view_model = YOLO(view_model_path) if view_model_path and os.path.exists(view_model_path) else YOLO('yolov8n.pt')
        self.dim_model = YOLO(dim_model_path) if dim_model_path and os.path.exists(dim_model_path) else YOLO('yolov8n.pt')
        
        # OCR reader
        self.ocr_reader = None
        if EASYOCR_AVAILABLE:
            self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            
        # --- PATCH 3: Define OCR trigger IDs based on new classes ---
        # Assuming mapping: 0=dimension, 1=tolerance_upper, 2=tolerance_lower, 3=tolerance
        self.TEXT_CLASS_IDS = [0, 1, 2, 3]
        self.DIM_CLASSES = ['dimension', 'tolerance_upper', 'tolerance_lower', 'tolerance']
        # --- END PATCH 3 ---
    
    def train_view_detector(self, view_yaml: str, epochs: int = 100, 
                            imgsz: int = 640, batch: int = 16):
        """Train the VIEW detection model"""
        print("\n" + "=" * 60)
        print("Training VIEW Detection Model")
        print("=" * 60)
        
        results = self.view_model.train(
            data=view_yaml,
            # PATCH 5: Set views training epochs to 100
            epochs=epochs, 
            imgsz=imgsz,
            batch=batch,
            project='runs/detect',
            name='view_detector',
            patience=20,
            save=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=True
        )
        
        print("✓ View detector training complete!")
        return results
    
    def train_dimension_detector(self, dim_yaml: str, epochs: int = 100,
                                 imgsz: int = 640, batch: int = 16):
        """Train the DIMENSION detection model"""
        print("\n" + "=" * 60)
        print("Training DIMENSION Detection Model")
        print("=" * 60)
        
        results = self.dim_model.train(
            data=dim_yaml,
            # PATCH 5: Set dimension training epochs to 200
            epochs=epochs, 
            # PATCH 5: Set dimension image size to 800
            imgsz=imgsz,
            batch=batch,
            project='runs/detect',
            name='dimension_detector',
            patience=20,
            save=True,
            device=0 if torch.cuda.is_available() else 'cpu',
            plots=True
        )
        
        print("✓ Dimension detector training complete!")
        return results
    
    def detect_views(self, image_path: str, conf: float = 0.25):
        """Detect views using the view model"""
        results = self.view_model.predict(image_path, conf=conf, save=False)
        return results[0]
    
    def detect_dimensions(self, image: np.ndarray, conf: float = 0.25):
        """Detect dimensions using the dimension model"""
        results = self.dim_model.predict(image, conf=conf, save=False)
        return results[0]
    
    def full_pipeline(self, image_path: str, output_dir: str = './output'):
        """
        Complete two-stage pipeline:
        1. Detect and crop views
        2. Detect dimensions in each view
        3. Extract text with OCR (with pre-processing for robustness)
        4. Generate annotated outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("Running Full Two-Stage Pipeline")
        print("=" * 60)
        
        # Load image
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Stage 1: Detect views
        print("\n[Stage 1] Detecting views...")
        view_results = self.detect_views(image_path)
        views = []
        
        for i, box in enumerate(view_results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            views.append({
                'id': i,
                'bbox': (x1, y1, x2, y2),
                'class': class_id,
                'confidence': conf,
                'crop': img[y1:y2, x1:x2]
            })
        
        print(f"  Found {len(views)} views")
        
        # Stage 2: Detect dimensions in each view
        print("\n[Stage 2] Detecting dimensions in each view...")
        all_dimensions = []
        
        for i, view in enumerate(views):
            print(f"  Processing view {i+1}/{len(views)}...")
            crop = view['crop']
            
            # Detect dimensions in cropped view
            dim_results = self.detect_dimensions(crop)
            
            # Extract text with OCR
            dimensions = []
            for box in dim_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                
                # Check if the class ID corresponds to one of the text-bearing features
                if class_id in self.TEXT_CLASS_IDS and self.ocr_reader:
                    roi = crop[y1:y2, x1:x2]
                    
                    # --- OCR IMPROVEMENT 1: Image Pre-processing (Binarization) ---
                    try:
                        # Convert to grayscale
                        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        # Apply Otsu's thresholding for clean black/white image
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        ocr_input = thresh
                    except cv2.error:
                        ocr_input = roi
                        # Warning is useful for debugging bad crops
                        print(f"    Warning: Binarization failed for view {i}, class {self.DIM_CLASSES[class_id]} at ({x1}, {y1}). Using raw crop.")
                    
                    ocr_results = self.ocr_reader.readtext(ocr_input)
                    # -------------------------------------------------------------
                    
                    for (bbox, text, conf) in ocr_results:
                        # --- OCR IMPROVEMENT 2: Lowered Confidence Threshold (0.2) ---
                        if conf > 0.2: 
                            dimensions.append({
                                'class_name': self.DIM_CLASSES[class_id], # Include class name for clarity
                                'text': text,
                                'bbox': (x1, y1, x2, y2),
                                'confidence': conf,
                                'view_id': i
                            })
                        # ------------------------------------------------------
            
            view['dimensions'] = dimensions
            all_dimensions.extend(dimensions)
            
            # Save annotated crop
            annotated_crop = dim_results.plot()
            cv2.imwrite(f'{output_dir}/view_{i}_annotated.png', annotated_crop)
            cv2.imwrite(f'{output_dir}/view_{i}_original.png', crop)
            
        print(f"  Found {len(all_dimensions)} dimensions total")
        
        # Stage 3: Generate final annotated image
        print("\n[Stage 3] Generating final annotated image...")
        annotated = img.copy()
        
        # Draw views
        for view in views:
            x1, y1, x2, y2 = view['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(annotated, f"View {view['id']}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        cv2.imwrite(f'{output_dir}/final_annotated.png', annotated)
        
        # Save results summary
        with open(f'{output_dir}/results.txt', 'w') as f:
            f.write("DETECTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Views: {len(views)}\n")
            f.write(f"Total Dimensions: {len(all_dimensions)}\n\n")
            
            for view in views:
                f.write(f"\nView {view['id']}:\n")
                f.write(f"  Position: {view['bbox']}\n")
                f.write(f"  Dimensions found: {len(view['dimensions'])}\n")
                for dim in view['dimensions']:
                    # Write the extracted class name alongside the text
                    f.write(f"    - [{dim['class_name']}] {dim['text']} (conf: {dim['confidence']:.2f})\n")
        
        print(f"\n✓ Pipeline complete! Results saved to {output_dir}/")
        
        return {
            'views': views,
            'dimensions': all_dimensions
        }


def main():
    """Complete pipeline using BOTH datasets"""
    
    print("=" * 60)
    print("TWO-DATASET TECHNICAL DRAWING PIPELINE")
    print("=" * 60)
    
    # Paths to your two datasets
    VIEW_DATASET = './dataset_views'
    DIM_DATASET = './dataset_dimensions'
    
    # Step 1: Prepare both datasets
    print("\n[STEP 1] Preparing Both Datasets")
    manager = DualDatasetManager(VIEW_DATASET, DIM_DATASET)
    view_yaml, dim_yaml = manager.prepare_both_datasets()
    
    # Visualize both
    manager.visualize_both_datasets(num_samples=3)
    
    if not view_yaml or not dim_yaml:
        print("\nError: Dataset preparation failed. Check your dataset paths.")
        return
    
    # Step 2: Train both models
    print("\n[STEP 2] Training Both Models")
    detector = TwoStageDetector()
    
    # Train view detector
    # PATCH 5: Set epochs to 100 for view detector
    detector.train_view_detector(view_yaml, epochs=100, batch=8) 
    
    # Train dimension detector
    # PATCH 5: Set epochs to 200 and image size to 800 for dimension detector
    detector.train_dimension_detector(dim_yaml, epochs=200, imgsz=800, batch=8)
    
    # Step 3: Load trained models and run inference
    print("\n[STEP 3] Running Two-Stage Inference")
    detector = TwoStageDetector(
        view_model_path='runs/detect/view_detector/weights/best.pt',
        dim_model_path='runs/detect/dimension_detector/weights/best.pt'
    )
    
    # Test on a sample image
    test_image = 'test_image.png'  # Replace with your test image
    if os.path.exists(test_image):
        results = detector.full_pipeline(test_image)
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Views detected: {len(results['views'])}")
        print(f"Dimensions extracted: {len(results['dimensions'])}")


if __name__ == "__main__":
    print("Requirements Check:")
    print(f"  - PyTorch: {torch.cuda.is_available() and 'Available (GPU)' or 'Available (CPU)'}")
    print(f"  - YOLOv8: {YOLO_AVAILABLE and 'Installed' or 'NOT INSTALLED'}")
    print(f"  - EasyOCR: {EASYOCR_AVAILABLE and 'Installed' or 'NOT INSTALLED'}")
    
    if not YOLO_AVAILABLE:
        print("\nInstall: pip install ultralytics easyocr opencv-python matplotlib pyyaml")
    else:
        # Uncomment to run
        main()
        print("\nFull pipeline execution finished.")

