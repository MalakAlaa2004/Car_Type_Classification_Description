import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datasets import load_dataset

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
HF_DATASET_ID = "tanganke/stanford_cars"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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
# 3. DATASET WRAPPER
# ==========================================
class HFCarDataset(Dataset):
    """
    Wraps the Hugging Face dataset object to work with PyTorch.
    - Handles Grayscale -> RGB conversion.
    - Applies PyTorch transforms.
    """
    def __init__(self, hf_data, transform=None):
        self.data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        label = item['label']

        # Critical Cleaning Step: Some Stanford Cars images are Greyscale (1 channel).
        # Models expect RGB (3 channels). Convert if necessary.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# 4. MAIN LOADER FUNCTION
# ==========================================
def get_dataloaders(batch_size=32, img_size=224, num_workers=0):
    """
    Downloads data, creates splits, and returns PyTorch DataLoaders.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f" Loading '{HF_DATASET_ID}' from Hugging Face Hub...")
    dataset = load_dataset(HF_DATASET_ID)
    
    # 1. Create Splits
    # The dataset comes with 'train' and 'test'. 
    # We split 'train' further to create a Validation set.
    full_train = dataset['train']
    test_split = dataset['test']
    
    # 80% Train, 20% Validation
    split = full_train.train_test_split(test_size=0.2, seed=42)
    train_split = split['train']
    val_split = split['test']
    
    print(f"✅ Data Split: {len(train_split)} Train | {len(val_split)} Val | {len(test_split)} Test")

    # 2. Get Transforms
    tfms = get_transforms(img_size)

    # 3. Create Datasets
    train_ds = HFCarDataset(train_split, transform=tfms['train'])
    val_ds = HFCarDataset(val_split, transform=tfms['val'])
    test_ds = HFCarDataset(test_split, transform=tfms['val']) # Test uses same transform as Val

    # 4. Create Loaders
    # pin_memory=True speeds up transfer to GPU
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
