# CSCI611 — Assignment 2: Training and Visualizing a CNN on CIFAR-10

**Student:** Aditya Jekkula  
**Course:** CSCI 611 — Deep Learning  
**Repository:** CSCI611_Aditya_Jekkula  

---

## Overview

This assignment involves designing and training a custom Convolutional Neural Network (CNN) on the CIFAR-10 dataset, followed by visualizing its internal learned representations through feature maps and maximally activating images.

The project is split into two Jupyter notebooks:

| Notebook | Description |
|---|---|
| `Assignment_2_CNN_CIFAR10.ipynb` | Task 1 — CNN design, training, and evaluation |
| `Task2_Feature_Visualization.ipynb` | Task 2 — Feature map and activation visualization |

---

## Results Summary

| Metric | Value |
|---|---|
| Framework | TensorFlow 2.19 |
| Dataset | CIFAR-10 |
| Final Test Accuracy | ~85% |
| Epochs Trained | 50 |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Sparse Categorical Cross-Entropy |

---

## Repository Structure

```
CSCI611_Aditya_Jekkula/
└── Assignment_2/
    ├── Assignment_2_CNN_CIFAR10.ipynb      # Task 1: CNN training notebook
    ├── Task2_Feature_Visualization.ipynb   # Task 2: Visualization notebook
    ├── Assignment2_CNN_Report.pdf          # Final PDF report
    └── README.md                           # This file
```

---

## How to Run

### Option 1: Google Colab (Recommended — No Setup Required)

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Upload either `.ipynb` file from this repository
4. Enable GPU: **Runtime → Change runtime type → T4 GPU**
5. Click **Runtime → Run all**

> Both notebooks are self-contained. They will automatically download CIFAR-10, install any missing dependencies, train the model, and generate all output figures.

---

### Option 2: Run Locally

#### Prerequisites

- Python 3.8 or higher
- pip

#### Installation

```bash
# Clone the repository
git clone https://github.com/AdityaJekkula/CSCI611_Aditya_Jekkula.git
cd CSCI611_Aditya_Jekkula/Assignment_2

# Install dependencies
pip install tensorflow numpy matplotlib pillow jupyter
```

#### Run the notebooks

```bash
# Start Jupyter
jupyter notebook

# Then open either notebook in your browser:
# - Assignment_2_CNN_CIFAR10.ipynb
# - Task2_Feature_Visualization.ipynb
```

> **Note:** Training on CPU will take significantly longer (~30–60 min). GPU is strongly recommended.

---

## Notebook Details

### Task 1 — `Assignment_2_CNN_CIFAR10.ipynb`

Covers the full CNN pipeline:

- **Data loading & preprocessing** — CIFAR-10 normalized to [0,1], 90/10 train/val split
- **Data augmentation** — horizontal flip, shift ±10%, rotation ±15°, zoom 10%
- **Model architecture** — 3 convolutional blocks (6 Conv2D layers total), 3 MaxPooling layers, ReLU activations, BatchNormalization, Dropout
- **Training** — Adam optimizer, ReduceLROnPlateau + EarlyStopping callbacks
- **Evaluation** — test accuracy and loss on 10,000 held-out test images
- **Visualizations** — augmentation examples, training/validation loss and accuracy curves

All cell outputs and figures are saved in the notebook.

---

### Task 2 — `Task2_Feature_Visualization.ipynb`

Covers internal CNN visualization:

**Part A — Feature Map Visualization (First Conv Layer)**
- Selects 3 test images from different classes: airplane, cat, horse
- Extracts feature maps from `relu1_1` (first Conv2D activation)
- Visualizes 8 feature maps per image + all 32 in extended view
- Visualizes learned filter kernels from `conv1_1`

**Part B — Maximally Activating Images**
- Chosen layer: `relu2_2` (middle layer, Block 2)
- Chosen filters: Filter 0, Filter 10, Filter 25
- Activation metric: **mean** of the 16×16 feature map
- Finds top-5 test images per filter across all 10,000 test images
- Generates per-class activation bar charts and top-20 activation rankings

All cell outputs and figures are saved in the notebook.

---

## Model Architecture

```
Input (32 × 32 × 3)
│
├── Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
├── Block 3: Conv2D(128) → BN → ReLU → Conv2D(128) → BN → ReLU → MaxPool(2×2) → Dropout(0.25)
│
├── Flatten
├── Dense(256) → BN → ReLU → Dropout(0.5)
└── Dense(10) → Softmax
```

**Total Parameters:** 815,530 trainable

---

## Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | Sparse Categorical Cross-Entropy |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Batch Size | 64 |
| Max Epochs | 50 |
| LR Schedule | ReduceLROnPlateau (factor=0.5, patience=5) |
| Early Stopping | patience=10, restore best weights |
| Regularization | BatchNormalization + Dropout (0.25 / 0.5) |
| Augmentation | H-flip, shift ±10%, rotation ±15°, zoom 10% |

---

## Dependencies

| Package | Version |
|---|---|
| TensorFlow | 2.19+ |
| NumPy | any recent |
| Matplotlib | any recent |
| Pillow | any recent |
| Jupyter | any recent |

---

## Execution Traces

All code cells in both notebooks were fully executed in Google Colab with a T4 GPU. Cell outputs — including printed metrics, training logs, and all figures — are saved directly in the notebooks and visible without re-running.

---

## Contact

**Aditya Jekkula**  
CSCI 611 — Applied Machine Learning
California State University, Chico
