# DefectForge

**Few-Shot Industrial Anomaly Detection and Segmentation using DINOv2, PatchCore, FAISS, and SAM**

DefectForge is a computer vision project for detecting and localizing defects in industrial products using pretrained vision models and normal training examples.

The current implementation is evaluated on the screw category of the MVTec AD dataset.

## Features

- **DINOv2:** Extracts patch-level visual features from industrial images.
- **PatchCore-style memory bank:** Stores representative features from normal training images.
- **FAISS:** Finds anomalous patches through nearest-neighbor similarity search.
- **Segment Anything Model (SAM):** Generates candidate segmentation masks around suspicious regions.
- **OpenCV foreground filtering:** Restricts anomaly prompts to the detected product.
- **Gradio:** Provides an interactive interface for visualizing anomaly heatmaps and predicted defect masks.

## Architecture

```text
Input Screw Image
        |
        v
DINOv2 Feature Extraction
        |
        v
FAISS Nearest-Neighbor Search
        |
        v
Anomaly Heatmap
        |
        v
OpenCV Foreground Filtering
        |
        v
Five Anomaly Prompts
        |
        v
SAM (15 Candidate Masks)
        |
        v
Automatic Candidate Ranking
        |
        v
Final Predicted Defect Mask
```

## Dataset

MVTec Anomaly Detection — Screw category.

- 320 normal training images
- 41 normal test images
- 119 defective test images

Normal training images are used to construct the feature memory bank. Ground-truth defect masks are used for evaluation.

## Experimental Results

| Metric | Result |
|---|---:|
| Image-level AUROC | 0.9063 |
| Pixel-level AUROC | 0.9600 |
| SAM Mean IoU (baseline) | 0.4350 |
| SAM Mean IoU (foreground filtering) | 0.4493 |

### SAM Mean IoU by Defect Category

| Defect Category | Mean IoU |
|---|---:|
| Manipulated Front | 0.2681 |
| Scratch Head | 0.6331 |
| Scratch Neck | 0.7342 |
| Thread Side | 0.1020 |
| Thread Top | 0.4844 |

The results show that the pipeline performs better on relatively localized scratches than on certain small thread defects.

Foreground filtering improved overall mean IoU from 0.4350 to 0.4493.

These results are exploratory because development decisions were informed by the test set.

## Running the Demo

Create and activate a Python virtual environment, install the project dependencies, and prepare the pretrained model checkpoints and FAISS index.

Run:

```bash
python src/app.py
```

Open the local Gradio URL displayed in the terminal.

Upload a screw image to visualize the anomaly heatmap and automatically selected SAM mask.

## Limitations

- Currently evaluated only on the MVTec AD screw category.
- SAM may miss very small or fragmented defects.
- Foreground detection assumes a relatively distinguishable product and background.
- The automatic mask-ranking score is not a calibrated defect probability.
- The current implementation is a simplified PatchCore-style pipeline rather than an exact reproduction of the original PatchCore method.

## Future Work

- Diffusion-based synthetic defect generation
- Improved anomaly-guided SAM prompting
- Evaluation on additional industrial categories
- ONNX/TensorRT inference optimization

## Technology Stack

Python, PyTorch, Hugging Face Transformers, DINOv2, FAISS, Segment Anything, OpenCV, NumPy, scikit-learn, and Gradio.
