"""
PyTorch Dataset for LUNA16 preprocessed data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class LUNA16Dataset(Dataset):
    """
    Dataset class for preprocessed LUNA16 patches
    """
    def __init__(self, data_dir='../data/processed/', mode='train', train_split=0.7, val_split=0.15):
        """
        Args:
            data_dir: Directory with preprocessed data
            mode: 'train', 'val', or 'test'
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        
        # Load data
        print(f"Loading {mode} data...")
        positive_patches = np.load(self.data_dir / 'positive_patches.npy')
        positive_masks = np.load(self.data_dir / 'positive_masks.npy')
        negative_patches = np.load(self.data_dir / 'negative_patches.npy')
        
        # Combine positive and negative samples
        all_patches = np.concatenate([positive_patches, negative_patches], axis=0)
        all_masks = np.concatenate([positive_masks, np.zeros_like(negative_patches)], axis=0)
        
        # Create labels (1 for nodule, 0 for background)
        labels = np.concatenate([
            np.ones(len(positive_patches)),
            np.zeros(len(negative_patches))
        ])
        
        # Shuffle data
        indices = np.random.permutation(len(all_patches))
        all_patches = all_patches[indices]
        all_masks = all_masks[indices]
        labels = labels[indices]
        
        # Split data
        n_samples = len(all_patches)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        if mode == 'train':
            self.patches = all_patches[:train_end]
            self.masks = all_masks[:train_end]
            self.labels = labels[:train_end]
        elif mode == 'val':
            self.patches = all_patches[train_end:val_end]
            self.masks = all_masks[train_end:val_end]
            self.labels = labels[train_end:val_end]
        elif mode == 'test':
            self.patches = all_patches[val_end:]
            self.masks = all_masks[val_end:]
            self.labels = labels[val_end:]
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'train', 'val', or 'test'")
        
        print(f" Loaded {len(self.patches)} {mode} samples")
        print(f"   Positive: {int(self.labels.sum())}, Negative: {int(len(self.labels) - self.labels.sum())}")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
            patch: Input image (1, D, H, W)
            mask: Segmentation mask (1, D, H, W)
            label: Binary label (nodule or not)
        """
        patch = self.patches[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        
        # Add channel dimension and convert to torch tensors
        patch = torch.FloatTensor(patch).unsqueeze(0)  # (1, 64, 64, 64)
        mask = torch.FloatTensor(mask).unsqueeze(0)    # (1, 64, 64, 64)
        label = torch.FloatTensor([label])
        
        return patch, mask, label


def get_dataloaders(data_dir='../data/processed/', batch_size=4, num_workers=2):
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory with preprocessed data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = LUNA16Dataset(data_dir=data_dir, mode='train')
    val_dataset = LUNA16Dataset(data_dir=data_dir, mode='val')
    test_dataset = LUNA16Dataset(data_dir=data_dir, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing LUNA16Dataset...")
    print("=" * 60)
    
    # Create datasets
    train_dataset = LUNA16Dataset(mode='train')
    val_dataset = LUNA16Dataset(mode='val')
    test_dataset = LUNA16Dataset(mode='test')
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset) + len(test_dataset)}")
    
    # Test getting a sample
    patch, mask, label = train_dataset[0]
    print(f"\n{'='*60}")
    print("SAMPLE DATA")
    print(f"{'='*60}")
    print(f"Patch shape: {patch.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Label: {label.item()}")
    print(f"Patch value range: [{patch.min():.3f}, {patch.max():.3f}]")
    print(f"Mask value range: [{mask.min():.3f}, {mask.max():.3f}]")
    
    # Test dataloader
    print(f"\n{'='*60}")
    print("TESTING DATALOADER")
    print(f"{'='*60}")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=2, num_workers=0)
    
    batch_patches, batch_masks, batch_labels = next(iter(train_loader))
    print(f"Batch patches shape: {batch_patches.shape}")
    print(f"Batch masks shape: {batch_masks.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    print(f"\n Dataset and DataLoader working correctly!")