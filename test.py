from drawing_pipeline import TwoStageDetector   # update with actual filename

detector = TwoStageDetector(
   view_model_path='runs/detect/view_detector/weights/best.pt',
    dim_model_path='runs/detect/dimension_detector/weights/best.pt'
)

test_image = "testimg12.jpg"   # ← your new image file

results = detector.full_pipeline(test_image, output_dir="./output_new_test12")

print("Views:", len(results['views']))
print("Dimensions:", len(results['dimensions']))
