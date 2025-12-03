import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd

# ==========================================
# 1. STANDARDIZED PREPROCESSING
# ==========================================
def get_transforms(img_size=(224, 224)):
    """
    Returns the standard transforms for all models (ResNet, ViT, VGG).
    """
    # Standard ImageNet statistics
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    return {
        # Strong Augmentation for Training
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]),
        # Clean for Validation/Testing
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]),
        'test': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
    }

# ==========================================
# 2. UNIVERSAL DATASET CLASS
# ==========================================
class UniversalCarDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['filepath']
        label = int(row['label'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Fallback for corrupt images (black image)
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
