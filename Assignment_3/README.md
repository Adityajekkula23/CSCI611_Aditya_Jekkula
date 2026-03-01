# Assignment 3: Small Object Detection Using YOLO

**Course:** CSCI 611 – Computer Vision  
**Topic:** Traffic Sign Detection with YOLOv8 on the LISA Dataset

---

## Overview

This project implements a small object detection pipeline using YOLOv8 to detect traffic signs from vehicle-mounted camera footage. The pipeline evaluates a pre-trained YOLOv8n baseline (COCO weights) and fine-tunes it on the LISA Traffic Sign Dataset with configurations optimized for small object detection.

**Detected Classes:**
- `stopSign`
- `warning`
- `pedestrianCrossing`
- `signalAhead`

---

## Dataset

**LISA Traffic Sign Dataset** sourced from Roboflow Universe.  
Link: https://universe.roboflow.com/lisatrafficlight/lisa-traffic

The dataset contains ~9,800 real-world driving images pre-annotated in YOLOv8 format. No manual annotation was required.

---

## Requirements

All code is designed to run on **Google Colab** with a T4 GPU runtime. No local installation is needed.

Dependencies installed automatically in the notebook:
```
ultralytics
roboflow
opencv-python
numpy
torch
matplotlib
```

---

## How to Run

1. Open [Google Colab](https://colab.research.google.com)
2. Upload the notebook file `A3_Small_Object_Detection_YOLO.ipynb`
3. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
4. In **Section 2**, paste your Roboflow API download snippet into the designated cell
5. Run all cells top to bottom: **Runtime → Run all**

The notebook will automatically:
- Download and filter the LISA dataset to 4 classes
- Evaluate the pre-trained YOLOv8n baseline
- Fine-tune the model with small-object-optimized settings
- Generate all evaluation metrics, charts, and detection visualizations

---

## Notebook Structure

| Section | Description |
|---------|-------------|
| Section 1 | Environment setup and GPU verification |
| Section 2 | Dataset download, class filtering, test split creation |
| Section 3 | Baseline evaluation using pre-trained YOLOv8n (COCO) |
| Section 4 | Fine-tuning with small-object configurations |
| Section 5 | Post-training evaluation and model comparison |
| Section 6 | Inference visualization and confidence analysis |
| Section 7 | Confidence threshold and NMS optimization |
| Section 8 | Final summary and results |

---

## Key Training Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `imgsz` | 1280 | Higher resolution preserves small sign detail |
| `epochs` | 30 | Sufficient convergence with early stopping |
| `batch` | 8 | Accommodates larger resolution on T4 GPU |
| `augment` | True | Scale, flip, blur augmentations |
| `mosaic` | 1.0 | Multi-scale object exposure |
| `optimizer` | AdamW | Stable fine-tuning convergence |

---

## Results Summary

| Metric | Pre-trained (COCO) | Fine-tuned (LISA) |
|--------|-------------------|-------------------|
| mAP@0.50 | N/A | 0.4588 |
| Precision | N/A | 0.4828 |
| Recall | N/A | 0.4241 |
| Detections on test set | 2,427 (COCO classes) | 308 (LISA classes) |

---

## Repository Contents

```
Assignment_3/
├── A3_Small_Object_Detection_YOLO.ipynb   # Main notebook
├── A3_Report.pdf                          # Full assignment report
├── README.md                              # This file
```
