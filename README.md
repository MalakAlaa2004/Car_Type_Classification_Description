# Fine-Grained Car Type Classification using Deep Learning

## 🌟 Project Overview

This repository contains the implementation and comprehensive analysis for a deep learning project focused on **Fine-Grained Car Type Classification** using the challenging Stanford Cars Dataset. The project evaluates four distinct Convolutional Neural Network (CNN) architectures to compare their performance on this complex task.

### Project Goal

The objective is to accurately classify the make and model of cars across 196 distinct classes. This is a fine-grained classification problem, requiring models to distinguish between visually similar sub-categories (e.g., different trims or model years of the same car).

### Dataset

*   **Name:** Stanford Cars Dataset
*   **Source:** [Hugging Face Dataset Link](https://huggingface.co/datasets/tanganke/stanford_cars)
*   **Classes:** 196 unique car makes and models.
*   **Total Images:** 16,185 (8,144 training, 8,041 testing).

---

## 🧠 Model Architectures Implemented

Four models were implemented and evaluated according to specific requirements:

| Model | Implementation Strategy | Key Architectural Feature |
| :--- | :--- | :--- |
| **VGG-19** | Implemented **from scratch** with Batch Normalization. | Uniform 3x3 convolutional filters. |
| **ResNet-50** | **Transfer Learning** (Fine-Tuning) using ImageNet pre-trained weights. | Residual Blocks (Skip Connections). |
| **Inception V1 (GoogLeNet)** | **Transfer Learning** (Fine-Tuning) using ImageNet pre-trained weights. | Inception Modules (Parallel Convolutions). |
| **MobileNetV2** | **Transfer Learning** (Fine-Tuning) using ImageNet pre-trained weights. | Inverted Residual Blocks and Linear Bottlenecks. |

---

## 📊 Key Results (196 Classes)

The models were rigorously evaluated on the 196-class test set. The **ResNet-50** model demonstrated superior performance, highlighting the effectiveness of residual connections for deep, fine-grained feature extraction.

| Model | Accuracy | F1-Score (Macro) | AUC (Macro) |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | **84.06%** | **83.98%** | **99.84%** |
| MobileNetV2 | 63.82% | 63.82% | 99.22% |
| VGG-19 (Scratch) | 46.81% | 46.68% | 98.03% |
| Inception V1 | 38.49% | 38.62% | 96.49% |

A detailed comparative analysis, including results from 10-class and 20-class trials, is available in the full project documentation.

---

## 📁 Repository Structure

The repository is organized to clearly separate the code, data preprocessing, and evaluation results for different classification granularities.

```
/car_type_-Classification_Description-my_branch/
├── 196_classes/
│   ├── Data Preprocessing_196_classes.py  # Data loading, augmentation, and splitting
│   ├── VGG-19_196_classes.ipynb          # VGG-19 (from scratch) training notebook
│   ├── ResNet_196_classes.ipynb          # ResNet-50 (transfer learning) training notebook
│   ├── InceptionV1_196_classes.ipynb     # Inception V1 (transfer learning) training notebook
│   ├── MobileNetV2_196_classes.ipynb     # MobileNetV2 (transfer learning) training notebook
│   ├── Evaluation_196_classes.py         # Script to generate all metrics and visualizations
│   └── Evaluation Results_196_classes/   # All output metrics, confusion matrices, and plots
├── 20_classes/                           # Code and results for the 20-class subset
├── 10_classes/                           # Code and results for the 10-class subset
├── Project_Documentation_Updated.pdf      # The comprehensive academic report
└── README.md                             # This file
```

---

## 🚀 Setup and Execution

### Prerequisites

*   Python 3.8+
*   PyTorch
*   Torchvision
*   Hugging Face `datasets` library

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd car_type_-Classification_Description-my_branch
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file containing `torch`, `torchvision`, `datasets`, `pandas`, `numpy`, `matplotlib`, and `scikit-learn` is assumed.)*

### Running the Code

The project is structured around Jupyter Notebooks for model training and a dedicated Python script for final evaluation.

1.  **Data Preparation:** The data will be automatically downloaded and preprocessed when running the training notebooks.
2.  **Training:** Execute the desired training notebook (e.g., `196_classes/ResNet_196_classes.ipynb`) to train the model and save the weights.
3.  **Evaluation:** Run the evaluation script to generate all final metrics and visualizations:
    ```bash
    python 196_classes/Evaluation_196_classes.py
    ```

All generated results, including the final comparison tables and plots, will be saved in the respective `Evaluation Results_XXX_classes` directory.

---

## 📝 Full Documentation

For a detailed breakdown of the methodology, data preprocessing, architectural explanations (with diagrams), and in-depth comparative analysis, please refer to the full academic report:

*   **[Project_Documentation_Updated.pdf](Project_Documentation_Updated.pdf)**
