# 📐 YOLOv8-based-Semantic-Segmentation-of-Dimension-Text-and-Tolerances (DimSense)

DimSense is a multi-stage Deep Learning pipeline designed to **automatically detect, isolate, and semantically categorize engineering dimensions and tolerances** from complex technical drawings and blueprints.

It converts raw drawing images into **structured, manufacturing-ready data**, extracting:
- Nominal dimension values
- Upper and lower tolerance components
- Positional metadata for future geometric association

---

## ✨ Key Features & Innovations

| Capability | Description | Technical Implementation |
|----------|-------------|------------------------|
| **Semantic Localization** | Classifies each dimension component like nominal, tolerance upper/lower | YOLOv8 custom models with non-overlapping class annotation |
| **Two-Model Pipeline** | View detection and dimension detection separated | Optimized training per task for accuracy & speed |
| **OCR Enhancement** | Improved readability for precise text extraction | OpenCV Otsu Thresholding before OCR |
| **OBB-Ready Design** | Handles rotated engineering text (future upgrade) | Architecture aligned with YOLOv8-OBB |
| **High Resolution Training** | Captures fine numerical details & tolerances | imgsz=800, 200 epochs training regime |
| **View Label Filtering Support** | Filters sheet reference labels like A, B | Planned class expansion |

---

## 💻 Tech Stack

| Component | Technology |
|----------|------------|
| Object Detection | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch + Torchvision |
| OCR Engine | EasyOCR → Planned: PaddleOCR |
| Image Processing | OpenCV |
| Scripting Support | Python, NumPy, PyYAML |

---

## ⚙️ Installation & Setup

### 🔹 Prerequisites
- Python **3.8+**
- GPU recommended for training (CUDA)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

🔹 Install Dependencies

pip install ultralytics easyocr opencv-python matplotlib pyyaml numpy torch

📂 Data Requirements
A single unified annotation source, split into 2 datasets:

mkdir -p dataset_views/images dataset_views/labels
mkdir -p dataset_dimensions/images dataset_dimensions/labels
Class mapping:

0 - nominal_dimension
1 - tolerance_upper
2 - tolerance_lower
3 - tolerance_combined
Place test images in project root (e.g., test_image.png)

🧠 Training & Inference
▶️ Train Full Pipeline
 
python drawing_pipeline.py

⏳ Training Specs
Detector	Epochs	Resolution	Notes
View Detector	100	640	Lightweight early locator
Dimension Detector	200	800	Trained on fine text details

➡️ Early stopping enabled with patience=20

📊 Output Format
Results available in:

./output/
Output File	Description
final_annotated.png	Full sheet detection + class overlays
view_0_annotated.png	Cropped view-level detection
results.txt	Parsed dimensional values + semantic labels

Example:

dimension: 25.0 mm
tolerance_upper: +0.2
tolerance_lower: -0.1

🗺️ Roadmap
Upgrade	Goal
YOLOv8-OBB	Accurate detection on rotated engineering text
PaddleOCR	Superior accuracy on blueprint typography
Geometric Graph Linking	Auto-connect dimensions to CAD geometries
Data Export API	CSV / JSON output for CAD-CAM workflow

👤 Author
Ojas Panse
AI & Data Science Engineer
📧 Email: ojaspanse200@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/ojas-panse-2a8a80286/

If you build upon this work, please consider acknowledging the project.
