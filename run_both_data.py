# Save this block as the content of drawing_pipeline.py
# ... [rest of the DualDatasetManager and TwoStageDetector classes] ...

def main():
    """Complete pipeline using BOTH datasets"""
    
    print("=" * 60)
    print("TWO-DATASET TECHNICAL DRAWING PIPELINE")
    print("=" * 60)
    
    # Paths to your two datasets
    VIEW_DATASET = 'dataset_views'
    DIM_DATASET = 'dataset_dimensions'
    
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
    # Initialize detector instance
    detector = TwoStageDetector() 
    
    # Train view detector
    # Using the suggested 50 epochs for the quick start
    detector.train_view_detector(view_yaml, epochs=50, batch=8) 
    
    # Train dimension detector
    # Using the suggested 50 epochs for the quick start
    detector.train_dimension_detector(dim_yaml, epochs=50, batch=8)
    
    # Step 3: Load trained models and run inference
    print("\n[STEP 3] Running Two-Stage Inference")
    detector = TwoStageDetector(
        view_model_path='runs/detect/view_detector/weights/best.pt',
        dim_model_path='runs/detect/dimension_detector/weights/best.pt'
    )
    
    # Test on a sample image
    test_image = 'test_image.png' # MUST exist in the same directory
    output_directory = './results'
    if os.path.exists(test_image):
        results = detector.full_pipeline(test_image, output_dir=output_directory)
        
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Views detected: {len(results['views'])}")
        print(f"Dimensions extracted: {len(results['dimensions'])}")
        print(f"Results saved to: {output_directory}")


if __name__ == "__main__":
    print("Requirements Check:")
    # ... [requirements check code] ...
    
    if not YOLO_AVAILABLE:
        print("\nInstall: pip install ultralytics easyocr opencv-python matplotlib pyyaml")
    else:
        # This line is the final executable command
        main() 
        print("\nFull pipeline execution finished.")