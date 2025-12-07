import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             roc_curve, auc, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from datasets import load_dataset
from data_preprocessing import get_transforms, HFCarDataset
from torch.utils.data import DataLoader
import json
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
HF_DATASET_ID = "tanganke/stanford_cars"

# Define all models to evaluate
MODELS_CONFIG = {
    'ResNet50': {
        'path': 'resnet50_new_hugg_preproce.pth',
        'architecture': 'resnet50',
        'description': 'Deep residual network with 50 layers',
        'paper': 'To be done'
    },
    'VGG19': {
        'path': 'vgg19_hf_stanford_cars.pth',
        'architecture': 'vgg19',
        'description': 'Very Deep Convolutional Networks',
        'paper': 'To be done'
    },
    'Inception V1': {
        'path': 'inception_v1_stanford_cars.pth',
        'architecture': 'inception_v1',
        'description': 'GoogLeNet with inception modules',
        'paper': 'To be done'
    },
    'MobileNetV2': {
        'path': 'mobilenet_v2_stanford_cars.pth',
        'architecture': 'mobilenet_v2',
        'description': 'Lightweight mobile architecture',
        'paper': 'To be done'
    }
}

BATCH_SIZE = 32
NUM_CLASSES = 196
IMG_SIZE = 224
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose which split to evaluate on: 'val' or 'test'
EVAL_SPLIT = 'test'  # Change to 'val' for validation evaluation

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================
def get_model(architecture, num_classes=196):
    """Load model architecture with custom classifier"""
    if architecture == 'resnet50':
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    elif architecture == 'vgg19':
        # Use VGG19 with Batch Normalization (vgg19_bn)
        model = models.vgg19_bn(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    elif architecture == 'inception_v1':
        model = models.googlenet(num_classes=NUM_CLASSES,aux_logits=True, init_weights=False)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    elif architecture == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model

# ==========================================
# 3. COMPREHENSIVE EVALUATION
# ==========================================
def evaluate_single_model(model_name, model_config, dataloader, dataset_size, class_names, save_prefix):
    """Evaluate a single model with comprehensive metrics"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    
    # Load Model
    model_path = model_config['path']
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    
    print(f"Loading Model from {model_path}...")
    model = get_model(model_config['architecture'], num_classes=NUM_CLASSES)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    model.to(device)
    model.eval()
    
    # Inference
    print(f"Running inference on {dataset_size} images...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ==========================================
    # METRICS CALCULATION
    # ==========================================
    print("\n" + "="*60)
    print("      TEST RESULTS       ")
    print("="*60)
    
    results = {}
    
    # 1. Overall Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    results['accuracy'] = accuracy
    print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")
    
    # 2. Precision, Recall, F-Score (per class and averaged)
    precision, recall, fscore, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, fscore_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    results['precision_macro'] = precision_macro
    results['recall_macro'] = recall_macro
    results['fscore_macro'] = fscore_macro
    results['precision_weighted'] = precision_weighted
    results['recall_weighted'] = recall_weighted
    results['fscore_weighted'] = fscore_weighted
    
    print(f"\n📈 Macro-Averaged Metrics:")
    print(f"   Precision: {precision_macro:.4f}")
    print(f"   Recall: {recall_macro:.4f}")
    print(f"   F1-Score: {fscore_macro:.4f}")
    
    print(f"\n📈 Weighted-Averaged Metrics:")
    print(f"   Precision: {precision_weighted:.4f}")
    print(f"   Recall: {recall_weighted:.4f}")
    print(f"   F1-Score: {fscore_weighted:.4f}")
    
    # 3. Classification Report with Class Names
    print("\nGenerating Detailed Report...")
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        output_dict=True, 
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    
    print("\n🏆 Top 5 Best Classified Cars (by F1-Score):")
    best_classes = report_df.sort_values(by='f1-score', ascending=False).head(5)
    print(best_classes[['precision', 'recall', 'f1-score', 'support']])
    
    print("\n⚠️ Top 5 Worst Classified Cars (by F1-Score):")
    classes_only = report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    worst_classes = classes_only.sort_values(by='f1-score', ascending=True).head(5)
    print(worst_classes[['precision', 'recall', 'f1-score', 'support']])
    
    # Save detailed report
    report_df.to_csv(f"{OUTPUT_DIR}/{save_prefix}_classification_report.csv")
    
    # ==========================================
    # VISUALIZATIONS
    # ==========================================
    
    # 1. Confusion Matrix
    print("\n📊 Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Full confusion matrix (normalized)
    plt.figure(figsize=(20, 18))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, cmap='RdYlGn', annot=False, fmt='.2f', 
                cbar_kws={'label': 'Normalized Count'})
    plt.title(f'Confusion Matrix - {model_name} (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_confusion_matrix_full.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Subset confusion matrix (first 20 classes with actual names)
    print("Generating Confusion Matrix (First 20 classes with names)...")
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm[:20, :20], annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names[:20], yticklabels=class_names[:20],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name} (First 20 Classes)', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=11)
    plt.ylabel('Actual', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_confusion_matrix_subset.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Class Metrics Bar Chart
    print("📊 Generating Per-Class Metrics...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    class_indices = np.arange(NUM_CLASSES)
    
    axes[0].bar(class_indices, precision, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Per-Class Precision - {model_name}', fontsize=13, fontweight='bold')
    axes[0].axhline(y=precision_macro, color='r', linestyle='--', label=f'Macro Avg: {precision_macro:.3f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(class_indices, recall, color='seagreen', alpha=0.7)
    axes[1].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Per-Class Recall - {model_name}', fontsize=13, fontweight='bold')
    axes[1].axhline(y=recall_macro, color='r', linestyle='--', label=f'Macro Avg: {recall_macro:.3f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(class_indices, fscore, color='coral', alpha=0.7)
    axes[2].set_xlabel('Class Index', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[2].set_title(f'Per-Class F1-Score - {model_name}', fontsize=13, fontweight='bold')
    axes[2].axhline(y=fscore_macro, color='r', linestyle='--', label=f'Macro Avg: {fscore_macro:.3f}')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curves and AUC
    print("📊 Calculating ROC and AUC...")
    unique_classes = np.unique(all_labels)
    y_test_bin = label_binarize(all_labels, classes=range(NUM_CLASSES))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(NUM_CLASSES):
        if i in unique_classes:
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            roc_auc[i] = 0.0
    
    # Micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), all_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in unique_classes]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in unique_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(unique_classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    results['auc_micro'] = roc_auc["micro"]
    results['auc_macro'] = roc_auc["macro"]
    
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=3)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=3)
    
    # Plot some individual class curves
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(min(5, len(unique_classes))), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.6,
                 label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. AUC Distribution
    plt.figure(figsize=(12, 6))
    valid_aucs = [roc_auc[i] for i in unique_classes if roc_auc[i] > 0]
    plt.hist(valid_aucs, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(roc_auc["macro"], color='red', linestyle='--', linewidth=2, 
                label=f'Macro-Avg AUC: {roc_auc["macro"]:.3f}')
    plt.xlabel('AUC Score', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Classes', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of Per-Class AUC Scores - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_auc_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ {model_name} evaluation completed!")
    print(f"   Results saved to: {OUTPUT_DIR}/")
    
    return results

# ==========================================
# 4. COMPARATIVE ANALYSIS
# ==========================================
def compare_models(all_results):
    """Generate comprehensive comparison between all models"""
    print("\n" + "="*60)
    print("COMPARATIVE ANALYSIS - ALL MODELS")
    print("="*60)
    
    # Prepare data for comparison
    model_names = list(all_results.keys())
    metrics = ['accuracy', 'precision_macro', 'recall_macro', 'fscore_macro', 
               'auc_micro', 'auc_macro']
    
    # 1. Comparison Table
    comparison_data = []
    for model_name in model_names:
        row = [model_name]
        for metric in metrics:
            row.append(all_results[model_name].get(metric, 0))
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data, 
                                  columns=['Model'] + metrics)
    
    print("\n📊 Performance Comparison Table:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv(f"{OUTPUT_DIR}/model_comparison_table.csv", index=False)
    
    # 2. Bar Chart Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = [all_results[model][metric] * 100 for model in model_names]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(model_names)))
        
        bars = ax.bar(range(len(model_names)), values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)', fontweight='bold')
        ax.set_title(metric.replace('_', ' ').title(), fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_charts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Radar Chart for Overall Performance
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for model_name in model_names:
        values = [all_results[model_name][metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', '\n') for m in metrics], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Model Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison_radar.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Generate Comparison Report
    print("\n📝 Generating Detailed Comparison Report...")
    
    # Find best model for each metric
    best_models = {}
    for metric in metrics:
        best_model = max(model_names, key=lambda m: all_results[m][metric])
        best_models[metric] = (best_model, all_results[best_model][metric])
    
    print("\n🏆 Best Model per Metric:")
    for metric, (model, value) in best_models.items():
        print(f"   {metric.replace('_', ' ').title()}: {model} ({value*100:.2f}%)")
    
    # Save comprehensive report
    with open(f"{OUTPUT_DIR}/comparison_summary.txt", 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Models: {len(model_names)}\n")
        f.write(f"Models Evaluated: {', '.join(model_names)}\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*60 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("BEST MODEL PER METRIC:\n")
        f.write("-"*60 + "\n")
        for metric, (model, value) in best_models.items():
            f.write(f"{metric.replace('_', ' ').title()}: {model} ({value*100:.2f}%)\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("ARCHITECTURE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        for model_name in model_names:
            config = MODELS_CONFIG.get(model_name, {})
            f.write(f"\n{model_name}:\n")
            f.write(f"  Description: {config.get('description', 'N/A')}\n")
            f.write(f"  Reference: {config.get('paper', 'N/A')}\n")
            f.write(f"  Accuracy: {all_results[model_name]['accuracy']*100:.2f}%\n")
            f.write(f"  F1-Score (Macro): {all_results[model_name]['fscore_macro']*100:.2f}%\n")
            f.write(f"  AUC (Macro): {all_results[model_name]['auc_macro']*100:.2f}%\n")
    
    print(f"\n✅ Comparison report saved to: {OUTPUT_DIR}/comparison_summary.txt")

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION SYSTEM")
    print("Stanford Cars Dataset - 196 Classes")
    print("Using Hugging Face Data Loader")
    print("="*60)
    
    # Load data from Hugging Face (automatic from cache)
    print(f"\nLoading data from Hugging Face Hub (cache)...")
    print(f"Evaluation split: {EVAL_SPLIT.upper()}")
    
    try:
        # Load dataset - this will use cache if already downloaded
        hf_dataset = load_dataset(HF_DATASET_ID)
        
        # Get the actual class names
        class_names = hf_dataset['train'].features['label'].names
        print(f"Loaded {len(class_names)} class names.")
        
        # Prepare the evaluation split
        if EVAL_SPLIT == 'val':
            # Create validation split from train
            split = hf_dataset['train'].train_test_split(test_size=0.2, seed=42)
            eval_hf_data = split['test']
        elif EVAL_SPLIT == 'test':
            eval_hf_data = hf_dataset['test']
        else:
            raise ValueError(f"Invalid EVAL_SPLIT: {EVAL_SPLIT}. Use 'val' or 'test'.")
        
        # Create dataset and dataloader
        tfms = get_transforms(img_size=IMG_SIZE)
        eval_dataset = HFCarDataset(eval_hf_data, transform=tfms['val'])
        eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        dataset_size = len(eval_dataset)
        print(f"Dataset loaded: {dataset_size} images in {EVAL_SPLIT} split")
        
        all_results = {}
        
        # Evaluate each model
        for model_name, model_config in MODELS_CONFIG.items():
            save_prefix = model_name.lower().replace('-', '_').replace(' ', '_')
            results = evaluate_single_model(
                model_name, 
                model_config, 
                eval_dataloader, 
                dataset_size,
                class_names,
                save_prefix
            )
            
            if results:
                all_results[model_name] = results
        
        # Comparative analysis
        if len(all_results) > 1:
            compare_models(all_results)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print(f"All results saved to: {OUTPUT_DIR}/")
        print("="*60)
        
    except Exception as e:
        print("\n An error occurred during execution:")
        print(e)
        import traceback
        traceback.print_exc()