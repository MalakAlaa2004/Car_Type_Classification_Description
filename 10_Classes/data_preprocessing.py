import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset
import random

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
HF_DATASET_ID = "tanganke/stanford_cars"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 10  # Select 20 random classes
RANDOM_SEED = 42  # For reproducibility

# ==========================================
# 2. TRANSFORMS FACTORY
# ==========================================
def get_transforms(img_size=224):
    """
    Returns a dictionary of transforms for Train, Val, and Test.
    Args:
        img_size (int): Target image resolution (default 224 for ResNet/VGG).
                        Use 384 if training high-res ViT.
    """
    return {
        # Strong Augmentation for Training (Prevents Overfitting)
        'train': transforms.Compose([
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))), # Resize slightly larger
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),      # Random crop
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ]),
        # Clean Preprocessing for Validation/Testing
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    }

# ==========================================
# 3. CLASS SELECTION FUNCTION
# ==========================================
def select_random_classes(dataset, num_classes=20, seed=42):
    """
    Randomly selects a subset of classes from the dataset.
    
    Args:
        dataset: HuggingFace dataset split
        num_classes: Number of classes to randomly select
        seed: Random seed for reproducibility
    
    Returns:
        filtered_dataset: Dataset containing only selected classes
        selected_classes: List of selected class indices
        label_mapping: Dictionary mapping old labels to new labels (0 to num_classes-1)
    """
    # Get all unique labels in the dataset
    all_labels = set(dataset['label'])
    all_labels_list = sorted(list(all_labels))
    
    print(f"📊 Total classes in dataset: {len(all_labels_list)}")
    
    # Set seed for both random and torch for complete reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Randomly select num_classes
    selected_classes = sorted(random.sample(all_labels_list, min(num_classes, len(all_labels_list))))
    
    print(f"🎯 Selected {len(selected_classes)} random classes: {selected_classes}")
    
    # Create mapping from old labels to new labels (0 to num_classes-1)
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}
    
    # Filter dataset to only include selected classes
    filtered_dataset = dataset.filter(lambda x: x['label'] in selected_classes)
    
    print(f"✅ Filtered dataset size: {len(filtered_dataset)} samples")
    
    return filtered_dataset, selected_classes, label_mapping

# ==========================================
# 4. DATASET WRAPPER
# ==========================================
class HFCarDataset(Dataset):
    """
    Wraps the Hugging Face dataset object to work with PyTorch.
    - Handles Grayscale -> RGB conversion.
    - Applies PyTorch transforms.
    - Remaps labels to 0-indexed range.
    """
    def __init__(self, hf_data, transform=None, label_mapping=None):
        self.data = hf_data
        self.transform = transform
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']
        
        # Remap label if mapping is provided
        if self.label_mapping is not None:
            label = self.label_mapping[label]

        # Critical Cleaning Step: Some Stanford Cars images are Greyscale (1 channel).
        # Models expect RGB (3 channels). Convert if necessary.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 5. MAIN LOADER FUNCTION
# ==========================================
def get_dataloaders(batch_size=32, img_size=224, num_workers=0, num_classes=NUM_CLASSES, seed=RANDOM_SEED):
    """
    Downloads data, selects random classes, creates splits, and returns PyTorch DataLoaders.
    
    Args:
        batch_size: Batch size for DataLoaders
        img_size: Image resolution
        num_workers: Number of workers for DataLoader
        num_classes: Number of random classes to select
        seed: Random seed for class selection
    
    Returns:
        train_loader, val_loader, test_loader, selected_classes, label_mapping
    """
    print(f"🚀 Loading '{HF_DATASET_ID}' from Hugging Face Hub...")
    dataset = load_dataset(HF_DATASET_ID)
    
    # 1. Select Random Classes from Full Dataset
    full_train = dataset['train']
    test_split = dataset['test']
    
    # Filter train and test to selected classes
    filtered_train, selected_classes, label_mapping = select_random_classes(
        full_train, num_classes=num_classes, seed=seed
    )
    filtered_test, _, _ = select_random_classes(
        test_split, num_classes=len(selected_classes), seed=seed
    )
    # Use same selected classes for test set
    filtered_test = test_split.filter(lambda x: x['label'] in selected_classes)
    
    # 2. Create Train/Val Split (80/20)
    split = filtered_train.train_test_split(test_size=0.2, seed=seed)
    train_split = split['train']
    val_split = split['test']
    
    print(f"✅ Data Split: {len(train_split)} Train | {len(val_split)} Val | {len(filtered_test)} Test")
    print(f"📌 Classes remapped to range: 0-{len(selected_classes)-1}")

    # 3. Get Transforms
    tfms = get_transforms(img_size)

    # 4. Create Datasets with Label Mapping
    train_ds = HFCarDataset(train_split, transform=tfms['train'], label_mapping=label_mapping)
    val_ds = HFCarDataset(val_split, transform=tfms['val'], label_mapping=label_mapping)
    test_ds = HFCarDataset(filtered_test, transform=tfms['val'], label_mapping=label_mapping)

    # 5. Create Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, selected_classes, label_mapping