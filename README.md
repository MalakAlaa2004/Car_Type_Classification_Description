# 🚗 Fine-Grained Car Classification with Deep Learning

## 🌟 Project Overview

This repository presents a comprehensive deep learning solution for **Fine-Grained Car Type Classification** using the challenging **Stanford Cars Dataset**. The core objective is to accurately distinguish between 196 visually similar car makes and models (e.g., different trims or model years), a task that demands highly discriminative feature extraction.

The project implements and rigorously evaluates four state-of-the-art Convolutional Neural Network (CNN) architectures to compare their performance, scalability, and efficiency on this complex fine-grained task.

![Model Performance Comparison](https://private-us-east-1.manuscdn.com/sessionFile/TWR0UrijfBZrf1gzu3tpZb/sandbox/67WrMbnzH8rpKSrmnGV9bq-images_1765305921113_na1fn_L2hvbWUvdWJ1bnR1L2Nhcl9jbGFzc2lmaWNhdGlvbl9wcm9qZWN0L2RvY3VtZW50YXRpb25fZGlhZ3JhbXMvY29tcGFyaXNvbl8xOTZfY2xhc3Nlcw.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvVFdSMFVyaWpmQlpyZjFnenUzdHBaYi9zYW5kYm94LzY3V3JNYm56SDhycEtTcm1uR1Y5YnEtaW1hZ2VzXzE3NjUzMDU5MjExMTNfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTmhjbDlqYkdGemMybG1hV05oZEdsdmJsOXdjbTlxWldOMEwyUnZZM1Z0Wlc1MFlYUnBiMjVmWkdsaFozSmhiWE12WTI5dGNHRnlhWE52Ymw4eE9UWmZZMnhoYzNObGN3LnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=HUnG25WVCjv2mLHYAVW1TBhsJli~kSsmVQooiu6bS06W-iFWD51Q8oAEKLFRE2hwdWA1kBcuioSYVOAUj8KZBwAqpuFPC4-u27I6BS~sUa~rgmhxwIDQFfK9Ef673UfmcyHmjLTBfLSsXmd9AoojYMil8-k3mslgKgDd2auoUnIydVD4tTMNZntC9ckArxSifAuq8SlsJXcuSpa2pn0jwEfmzlqn9LQ7pgDc~Ee~s3Y4YUojnhaM~diMPsVCInOq6ljJ9JpDO6o6zJurTVUuiR02tGMPW1UxBnBWC7Q9kQxmuvG5QkbYh5VG8Dd8WmKEXjmZD8ry12Wma7bCS0L~Dg__)
*Figure: Comparative performance of the four models on the full 196-class dataset. ResNet-50 shows clear superiority.*

---

## 🧠 Model Architectures Implemented

Four distinct models were selected to represent different architectural philosophies and implementation strategies:

| Model | Implementation Strategy | Key Architectural Feature |
| :--- | :--- | :--- |
| **ResNet-50** | **Transfer Learning** (Fine-Tuning) | Residual Blocks (Skip Connections) to enable deep learning. |
| **Inception V1 (GoogLeNet)** | **Transfer Learning** (Fine-Tuning) | Inception Modules for multi-scale feature extraction. |
| **MobileNetV2** | **Transfer Learning** (Fine-Tuning) | Inverted Residual Blocks for high efficiency and low latency. |
| **VGG-19** | Implemented **from scratch** with Batch Normalization. | Uniform 3x3 convolutional filters to test depth importance. |

---

## 📊 Key Results and Comparative Analysis

The models were evaluated across three complexity levels (10, 20, and 196 classes). The **ResNet-50** model, leveraging deep residual learning and ImageNet pre-trained weights, achieved the highest performance on the full dataset.

### Performance on 196 Classes (Full Dataset)

| Model | Accuracy | F1-Score (Macro) | AUC (Macro) |
| :--- | :--- | :--- | :--- |
| **ResNet-50** | **84.06%** | **83.98%** | **99.84%** |
| MobileNetV2 | 63.82% | 63.82% | 99.22% |
| VGG-19 (Scratch) | 46.81% | 46.68% | 98.03% |
| Inception V1 | 38.49% | 38.62% | 96.49% |

### Conclusion

The superior performance of **ResNet-50** is attributed to its robust ability to learn highly discriminative features for fine-grained classification, demonstrating excellent scalability and stability as the number of classes increases.

---

## 📁 Repository Structure

The project is organized into modular directories based on the classification complexity, ensuring clear separation of code and results.

```
/car_type_Classification_Description/
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
├── Car_Classification_Documentation.pdf  # The comprehensive academic report (Final Deliverable)
├── Project Requirements/                 # Original project requirement documents
└── README.md                             # This file
```

---

## 🚀 Setup and Execution

### Prerequisites

*   Python 3.8+
*   PyTorch & Torchvision
*   Hugging Face `datasets` library
*   Standard scientific computing libraries (`pandas`, `numpy`, `matplotlib`, `scikit-learn`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/MalakAlaa2004/Car_Type_Classification_Description.git
    cd car_type_Classification_Description
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure a `requirements.txt` file is present in the root directory listing all dependencies.)*

### Running the Code

The project uses Jupyter Notebooks for training and a dedicated script for final evaluation.

1.  **Training:** Open and execute the desired training notebook (e.g., `196_classes/ResNet_196_classes.ipynb`) to train the model and save the weights.
2.  **Evaluation:** Run the evaluation script to generate all final metrics and visualizations:
    ```bash
    python 196_classes/Evaluation_196_classes.py
    ```
    All generated results are saved in the respective `Evaluation Results_XXX_classes` directory.

---

## 📝 Full Academic Documentation

For a detailed breakdown of the methodology, architectural explanations (with diagrams and citations), in-depth comparative analysis, and discussion on the 10-class and 20-class trials, please refer to the final academic report:

*   **[Car_Classification_Documentation.pdf](Car_Classification_Documentation.pdf)**

