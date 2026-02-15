"""
Utility functions for training and evaluation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class DiceLoss(torch.nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, predictions, targets):
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class CombinedLoss(torch.nn.Module):
    """
    Combination of Dice Loss and Binary Cross Entropy
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = torch.nn.BCELoss()
    
    def forward(self, predictions, targets):
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        return self.dice_weight * dice + self.bce_weight * bce


def dice_coefficient(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate Dice coefficient for evaluation
    
    Args:
        predictions: Model predictions (0-1)
        targets: Ground truth masks (0-1)
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        dice_score: Dice coefficient
    """
    predictions = (predictions > threshold).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    dice = (2. * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )
    
    return dice.item()


def iou_score(predictions, targets, threshold=0.5, smooth=1.0):
    """
    Calculate Intersection over Union (IoU)
    
    Args:
        predictions: Model predictions (0-1)
        targets: Ground truth masks (0-1)
        threshold: Threshold for binarization
        smooth: Smoothing factor
        
    Returns:
        iou: IoU score
    """
    predictions = (predictions > threshold).float()
    targets = targets.float()
    
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f" Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        filepath: Path to checkpoint
        
    Returns:
        epoch: Epoch number
        loss: Loss value
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f" Checkpoint loaded: {filepath}")
    return epoch, loss


def visualize_prediction(image, mask, prediction, slice_idx=32):
    """
    Visualize a single prediction
    
    Args:
        image: Input image (64, 64, 64)
        mask: Ground truth mask (64, 64, 64)
        prediction: Model prediction (64, 64, 64)
        slice_idx: Which slice to visualize
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask[slice_idx], cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(prediction[slice_idx], cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(train_losses, val_losses, train_dice, val_dice, save_path=None):
    """
    Plot training history
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_dice: List of training Dice scores
        val_dice: List of validation Dice scores
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Dice score plot
    axes[1].plot(train_dice, label='Train Dice', linewidth=2)
    axes[1].plot(val_dice, label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Score', fontsize=12)
    axes[1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Training history saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test Dice Loss
    predictions = torch.rand(2, 1, 64, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64, 64)).float()
    
    dice_loss = DiceLoss()
    loss = dice_loss(predictions, targets)
    print(f" Dice Loss: {loss.item():.4f}")
    
    combined_loss = CombinedLoss()
    loss = combined_loss(predictions, targets)
    print(f" Combined Loss: {loss.item():.4f}")
    
    dice = dice_coefficient(predictions[0, 0], targets[0, 0])
    print(f" Dice Coefficient: {dice:.4f}")
    
    iou = iou_score(predictions[0, 0], targets[0, 0])
    print(f"IoU Score: {iou:.4f}")
    
    print("\n All utility functions working!")