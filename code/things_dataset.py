"""
ThingsEEG Dataset for fine-tuning EEG → Image models.

Loads preprocessed ThingsEEG .pth files (from preprocess_things_eeg.py)
and provides paired (EEG, image) samples compatible with the existing
DreamDiffusion training pipeline.

Usage:
    from things_dataset import ThingsEEGDataset, create_things_dataset
    
    train_ds, test_ds = create_things_dataset(
        processed_dir='../dreamdiffusion/datasets/things_eeg_processed',
        subjects=[1, 2, 3, 4, 5],
        image_size=512,
        test_ratio=0.1,
    )
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class ThingsEEGDataset(Dataset):
    """
    Dataset that loads preprocessed ThingsEEG data.
    
    Each item returns:
        {
            'eeg': Tensor [n_channels, time_len],    # e.g. [63, 512]
            'image': Tensor [3, H, W],                # normalized [-1, 1]
            'label': int,                              # concept ID (0-1853)
            'object': str,                             # concept name
            'image_path': str,                         # full image path
        }
    
    Compatible with EEGtoImageSDXL.finetune() which expects:
        item['eeg'] and item['image']
    """
    
    def __init__(self, data_list, image_size=512, augment=False, 
                 n_channels=63, time_len=512):
        """
        Args:
            data_list: List of dicts from preprocessed .pth files
            image_size: Target image size (default: 512 for SDXL)
            augment: Whether to apply data augmentation
            n_channels: Expected number of EEG channels
            time_len: Expected number of time points
        """
        self.data = data_list
        self.image_size = image_size
        self.augment = augment
        self.n_channels = n_channels
        self.time_len = time_len
        
        # Image transform: resize + normalize to [-1, 1]
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        
        # Collect metadata
        self.labels = sorted(set(item['label'] for item in self.data))
        self.objects = sorted(set(item['object'] for item in self.data))
        self.num_classes = len(self.labels)
        
        print(f"ThingsEEGDataset: {len(self.data)} samples, "
              f"{self.num_classes} concepts, image_size={image_size}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # EEG: [n_channels, time_len]
        eeg = item['eeg']
        
        # Pad/truncate channels if needed
        if eeg.shape[0] < self.n_channels:
            pad = torch.zeros(self.n_channels - eeg.shape[0], eeg.shape[1])
            eeg = torch.cat([eeg, pad], dim=0)
        elif eeg.shape[0] > self.n_channels:
            eeg = eeg[:self.n_channels]
        
        # Load and transform image
        image_path = item['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            'eeg': eeg,                     # [n_channels, time_len]
            'image': image,                 # [3, H, W] in [-1, 1]
            'label': item['label'],         # int
            'object': item['object'],       # str
            'image_path': image_path,       # str
        }


def create_things_dataset(processed_dir, subjects=None, image_size=512,
                         test_ratio=0.1, seed=42, augment_train=True,
                         n_channels=63, time_len=512):
    """
    Create train and test datasets from preprocessed ThingsEEG data.
    
    Splits by unique concepts: test set contains held-out concepts
    to evaluate generalization to unseen categories.
    
    Args:
        processed_dir: Directory containing preprocessed .pth files
        subjects: List of subject IDs to include (default: all found)
        image_size: Target image size
        test_ratio: Fraction of concepts held out for testing
        seed: Random seed for reproducible split
        augment_train: Apply augmentation to training data
        n_channels: Number of EEG channels
        time_len: Number of time points per epoch
    
    Returns:
        train_dataset, test_dataset: ThingsEEGDataset instances
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Find all processed files
    all_data = []
    files_found = []
    
    if subjects is None:
        # Auto-detect subjects from files
        for f in sorted(os.listdir(processed_dir)):
            if f.endswith('_things_eeg.pth'):
                files_found.append(f)
    else:
        for sub_id in subjects:
            f = f'sub-{sub_id:02d}_things_eeg.pth'
            if os.path.exists(os.path.join(processed_dir, f)):
                files_found.append(f)
            else:
                print(f"WARNING: {f} not found in {processed_dir}")
    
    print(f"\nLoading {len(files_found)} subject files from {processed_dir}...")
    
    for f in files_found:
        path = os.path.join(processed_dir, f)
        saved = torch.load(path)
        data = saved['dataset']
        print(f"  {f}: {len(data)} epochs, {saved.get('n_channels', '?')} channels")
        all_data.extend(data)
    
    print(f"Total: {len(all_data)} epochs")
    
    if len(all_data) == 0:
        raise ValueError(f"No data found! Check {processed_dir}")
    
    # Split by concept to test generalization
    all_concepts = sorted(set(item['object'] for item in all_data))
    n_test_concepts = max(1, int(len(all_concepts) * test_ratio))
    
    random.shuffle(all_concepts)
    test_concepts = set(all_concepts[:n_test_concepts])
    train_concepts = set(all_concepts[n_test_concepts:])
    
    train_data = [item for item in all_data if item['object'] in train_concepts]
    test_data = [item for item in all_data if item['object'] in test_concepts]
    
    print(f"\nSplit: {len(train_concepts)} train concepts, "
          f"{len(test_concepts)} test concepts")
    print(f"  Train: {len(train_data)} epochs")
    print(f"  Test:  {len(test_data)} epochs")
    
    train_dataset = ThingsEEGDataset(
        train_data, image_size=image_size, augment=augment_train,
        n_channels=n_channels, time_len=time_len,
    )
    test_dataset = ThingsEEGDataset(
        test_data, image_size=image_size, augment=False,
        n_channels=n_channels, time_len=time_len,
    )
    
    return train_dataset, test_dataset