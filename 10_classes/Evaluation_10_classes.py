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
from data_preprocessing import get_dataloaders
import json
from datetime import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================

# Define all models to evaluate
MODELS_CONFIG = {
    'ResNet50': {
        'path': 'resnet50_stanford_cars_10classes.pth',
        'architecture': 'resnet50',
        'description': 'Deep residual network with 50 layers',
        'img_size': 224
    },
    'VGG19 (Scratch)': {
        'path': 'vgg19_scratch_stanford_cars_10classes.pth',
        'architecture': 'vgg19_scratch',
        'description': 'VGG-19 trained from scratch with BatchNorm',
        'img_size': 224
    },
    'Inception V1': {
        'path': 'inception_v1_stanford_cars_10classes.pth',
        'architecture': 'inception_v1',
        'description': 'GoogLeNet with inception modules',
        'img_size': 299
    },
    'MobileNetV2': {
        'path': 'mobilenet_v2_stanford_cars_10classes.pth',
        'architecture': 'mobilenet_v2',
        'description': 'Lightweight mobile architecture',
        'img_size': 224
    }
}

BATCH_SIZE = 32
NUM_CLASSES = 10  # Updated to 20 classes
OUTPUT_DIR = "evaluation_results_10classes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose which split to evaluate on: 'val' or 'test'
EVAL_SPLIT = 'test'  # Change to 'val' for validation evaluation

# ==========================================
# 2. MODEL ARCHITECTURES
# ==========================================
def get_model(architecture, num_classes=10):
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
    elif architecture == 'vgg19_scratch':
        # VGG-19 From Scratch architecture (same as training)
        class VGG19_Scratch(nn.Module):
            def __init__(self, num_classes=20):
                super(VGG19_Scratch, self).__init__()
                
                def conv_block(in_ch, out_ch):
                    return nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )

                self.features = nn.Sequential(
                    conv_block(3, 64), conv_block(64, 64),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    conv_block(64, 128), conv_block(128, 128),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    conv_block(128, 256), conv_block(256, 256), conv_block(256, 256), conv_block(256, 256),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    conv_block(256, 512), conv_block(512, 512), conv_block(512, 512), conv_block(512, 512),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    conv_block(512, 512), conv_block(512, 512), conv_block(512, 512), conv_block(512, 512),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )

                self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(512, 1024),
                    nn.ReLU(True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.global_pool(x)
                x = self.classifier(x)
                return x
        
        model = VGG19_Scratch(num_classes=num_classes)
        
    elif architecture == 'inception_v1':
        model = models.googlenet(
        weights=models.GoogLeNet_Weights.IMAGENET1K_V1,
        aux_logits=True
        )

        # Replace FC head (same as training)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

        
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
        print(f"❌ Error: Model file {model_path} not found. Skipping...")
        return None
    
    print(f"Loading Model from {model_path}...")
    model = get_model(model_config['architecture'], num_classes=NUM_CLASSES)
    
    try:
        # Load checkpoint (handles both old and new format)
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
            print(f"   Best Val Acc: {checkpoint.get('val_acc', 0):.2f}%")
        else:
            model.load_state_dict(checkpoint)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
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
            
            # Handle Inception model which may return tuple (main_output, aux2, aux1)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Use only the main output
            
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
    print(f"      {EVAL_SPLIT.upper()} RESULTS       ")
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
    
    # 1. Confusion Matrix (Full - 20 classes with names)
    print("\n📊 Generating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Normalized confusion matrix
    plt.figure(figsize=(14, 12))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    plt.title(f'Confusion Matrix - {model_name} (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Raw counts confusion matrix
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name} (Counts)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_confusion_matrix_counts.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Class Metrics Bar Chart
    print("📊 Generating Per-Class Metrics...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    class_indices = np.arange(NUM_CLASSES)
    
    axes[0].bar(class_indices, precision, color='steelblue', alpha=0.7)
    axes[0].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[0].set_title(f'Per-Class Precision - {model_name}', fontsize=13, fontweight='bold')
    axes[0].axhline(y=precision_macro, color='r', linestyle='--', label=f'Macro Avg: {precision_macro:.3f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_xticks(class_indices)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    
    axes[1].bar(class_indices, recall, color='seagreen', alpha=0.7)
    axes[1].set_ylabel('Recall', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Per-Class Recall - {model_name}', fontsize=13, fontweight='bold')
    axes[1].axhline(y=recall_macro, color='r', linestyle='--', label=f'Macro Avg: {recall_macro:.3f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_xticks(class_indices)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    
    axes[2].bar(class_indices, fscore, color='coral', alpha=0.7)
    axes[2].set_xlabel('Class', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    axes[2].set_title(f'Per-Class F1-Score - {model_name}', fontsize=13, fontweight='bold')
    axes[2].axhline(y=fscore_macro, color='r', linestyle='--', label=f'Macro Avg: {fscore_macro:.3f}')
    axes[2].legend()
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_xticks(class_indices)
    axes[2].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    
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
    
    # Plot all individual class curves for 20 classes
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
    for i, color in zip(range(NUM_CLASSES), colors):
        if i in unique_classes:
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5, alpha=0.6,
                     label=f'{class_names[i][:20]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{save_prefix}_roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. AUC Distribution
    plt.figure(figsize=(12, 6))
    valid_aucs = [roc_auc[i] for i in unique_classes if roc_auc[i] > 0]
    plt.hist(valid_aucs, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
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
    fig.suptitle('Model Performance Comparison (20 Classes)', fontsize=16, fontweight='bold')
    
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
    ax.set_title('Multi-Metric Model Comparison (20 Classes)', fontsize=14, fontweight='bold', pad=20)
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
        f.write(f"Number of Classes: {NUM_CLASSES}\n")
        f.write(f"Number of Models: {len(model_names)}\n")
        f.write(f"Models Evaluated: {', '.join(model_names)}\n")
        f.write(f"Evaluation Split: {EVAL_SPLIT}\n\n")
        
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
            f.write(f"  Image Size: {config.get('img_size', 'N/A')}\n")
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
    print("Stanford Cars Dataset - 20 Random Classes")
    print("="*60)
    
    try:
        # Load data using the same preprocessing pipeline
        print(f"\n📦 Loading data from Hugging Face Hub...")
        print(f"Evaluation split: {EVAL_SPLIT.upper()}")
        
        # Get dataloaders - we'll use the appropriate split
        train_loader, val_loader, test_loader, selected_classes, label_mapping = get_dataloaders(
            batch_size=BATCH_SIZE,
            img_size=224,  # Default, will be adjusted per model if needed
            num_workers=0,
            num_classes=NUM_CLASSES,
            seed=42  # Same seed as training to get same classes
        )
        
        # Select the appropriate dataloader
        if EVAL_SPLIT == 'val':
            eval_dataloader = val_loader
        elif EVAL_SPLIT == 'test':
            eval_dataloader = test_loader
        else:
            raise ValueError(f"Invalid EVAL_SPLIT: {EVAL_SPLIT}. Use 'val' or 'test'.")
        
        dataset_size = len(eval_dataloader.dataset)
        
        # Get class names from the label mapping (reverse mapping)
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        
        # Load full dataset to get original class names
        from datasets import load_dataset
        hf_dataset = load_dataset("tanganke/stanford_cars")
        all_class_names = hf_dataset['train'].features['label'].names
        
        # Get the actual class names for our selected classes
        class_names = [all_class_names[reverse_mapping[i]] for i in range(NUM_CLASSES)]
        
        print(f"\n📊 Selected Classes ({NUM_CLASSES}):")
        for i, name in enumerate(class_names):
            print(f"   {i}: {name}")
        
        print(f"\n✅ Dataset loaded: {dataset_size} images in {EVAL_SPLIT} split")
        
        all_results = {}
        
        # Evaluate each model
        for model_name, model_config in MODELS_CONFIG.items():
            # Check if this model requires a different image size
            model_img_size = model_config.get('img_size', 224)
            
            # Reload data with correct image size if needed
            if model_img_size != 224:
                print(f"\n🔄 Reloading data with image size {model_img_size} for {model_name}...")
                _, model_val_loader, model_test_loader, _, _ = get_dataloaders(
                    batch_size=BATCH_SIZE,
                    img_size=model_img_size,
                    num_workers=0,
                    num_classes=NUM_CLASSES,
                    seed=42
                )
                model_eval_loader = model_test_loader if EVAL_SPLIT == 'test' else model_val_loader
                model_dataset_size = len(model_eval_loader.dataset)
            else:
                model_eval_loader = eval_dataloader
                model_dataset_size = dataset_size
            
            save_prefix = model_name.lower().replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')
            results = evaluate_single_model(
                model_name, 
                model_config, 
                model_eval_loader, 
                model_dataset_size,
                class_names,
                save_prefix
            )
            
            if results:
                all_results[model_name] = results
        
        # Comparative analysis
        if len(all_results) > 1:
            compare_models(all_results)
        elif len(all_results) == 1:
            print("\n⚠️ Only one model evaluated. Skipping comparative analysis.")
        else:
            print("\n❌ No models were successfully evaluated.")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print(f"All results saved to: {OUTPUT_DIR}/")
        print("="*60)
        
    except Exception as e:
        print("\n❌ An error occurred during execution:")
        print(e)
        import traceback
        traceback.print_exc()