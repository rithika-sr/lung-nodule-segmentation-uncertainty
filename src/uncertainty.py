"""
Uncertainty Quantification using Monte Carlo Dropout
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path


class MonteCarloDropout:
    """
    Monte Carlo Dropout for uncertainty estimation
    """
    def __init__(self, model, device, n_samples=20):
        """
        Args:
            model: Trained U-Net model with dropout layers
            device: Device to run inference on
            n_samples: Number of Monte Carlo samples
        """
        self.model = model
        self.device = device
        self.n_samples = n_samples
        
        # Enable dropout during inference
        self.model.eval()
        self.model.enable_dropout()
    
    def predict_with_uncertainty(self, x):
        """
        Get prediction with uncertainty estimate
        
        Args:
            x: Input tensor (B, C, D, H, W)
            
        Returns:
            mean_prediction: Mean of MC samples
            uncertainty: Variance of MC samples (epistemic uncertainty)
        """
        x = x.to(self.device)
        
        # Collect predictions from multiple forward passes
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                predictions.append(pred.cpu())
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (n_samples, B, C, D, H, W)
        
        # Calculate mean and variance
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0)
        
        return mean_prediction, uncertainty
    
    def evaluate_uncertainty(self, dataloader):
        """
        Evaluate uncertainty on entire dataset
        
        Args:
            dataloader: PyTorch dataloader
            
        Returns:
            results: Dictionary with predictions and uncertainties
        """
        all_images = []
        all_masks = []
        all_predictions = []
        all_uncertainties = []
        
        print(f"Evaluating with {self.n_samples} MC samples...")
        
        for patches, masks, labels in tqdm(dataloader, desc="Processing"):
            mean_pred, uncertainty = self.predict_with_uncertainty(patches)
            
            all_images.append(patches.cpu())
            all_masks.append(masks.cpu())
            all_predictions.append(mean_pred)
            all_uncertainties.append(uncertainty)
        
        results = {
            'images': torch.cat(all_images, dim=0),
            'masks': torch.cat(all_masks, dim=0),
            'predictions': torch.cat(all_predictions, dim=0),
            'uncertainties': torch.cat(all_uncertainties, dim=0)
        }
        
        return results


def visualize_uncertainty(image, mask, prediction, uncertainty, slice_idx=32, save_path=None):
    """
    Visualize prediction with uncertainty map
    
    Args:
        image: Input image (C, D, H, W)
        mask: Ground truth mask (C, D, H, W)
        prediction: Model prediction (C, D, H, W)
        uncertainty: Uncertainty map (C, D, H, W)
        slice_idx: Which slice to visualize
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Remove channel dimension for visualization
    image = image.squeeze()
    mask = mask.squeeze()
    prediction = prediction.squeeze()
    uncertainty = uncertainty.squeeze()
    
    # Top row: Input, Ground Truth, Prediction
    axes[0, 0].imshow(image[slice_idx], cmap='gray')
    axes[0, 0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask[slice_idx], cmap='gray')
    axes[0, 1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(prediction[slice_idx], cmap='gray')
    axes[0, 2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Bottom row: Uncertainty maps
    im1 = axes[1, 0].imshow(uncertainty[slice_idx], cmap='hot')
    axes[1, 0].set_title('Uncertainty Map', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
    
    # Overlay uncertainty on prediction
    axes[1, 1].imshow(prediction[slice_idx], cmap='gray', alpha=0.7)
    im2 = axes[1, 1].imshow(uncertainty[slice_idx], cmap='hot', alpha=0.5)
    axes[1, 1].set_title('Prediction + Uncertainty Overlay', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
    
    # Thresholded high uncertainty regions
    high_uncertainty = (uncertainty[slice_idx] > uncertainty[slice_idx].mean())
    axes[1, 2].imshow(image[slice_idx], cmap='gray')
    axes[1, 2].contour(high_uncertainty, colors='red', linewidths=2)
    axes[1, 2].set_title('High Uncertainty Regions (Red)', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Uncertainty visualization saved: {save_path}")
    
    return fig


def analyze_uncertainty_statistics(uncertainties, predictions, masks):
    """
    Analyze uncertainty statistics
    
    Args:
        uncertainties: Uncertainty maps
        predictions: Model predictions
        masks: Ground truth masks
        
    Returns:
        stats: Dictionary with statistics
    """
    uncertainties = uncertainties.numpy()
    predictions = predictions.numpy()
    masks = masks.numpy()
    
    # Binarize predictions
    pred_binary = (predictions > 0.5).astype(float)
    
    # Calculate metrics
    correct_pixels = (pred_binary == masks)
    incorrect_pixels = ~correct_pixels
    
    # Uncertainty on correct vs incorrect predictions
    uncertainty_correct = uncertainties[correct_pixels]
    uncertainty_incorrect = uncertainties[incorrect_pixels]
    
    stats = {
        'mean_uncertainty': uncertainties.mean(),
        'std_uncertainty': uncertainties.std(),
        'max_uncertainty': uncertainties.max(),
        'min_uncertainty': uncertainties.min(),
        'mean_uncertainty_correct': uncertainty_correct.mean() if len(uncertainty_correct) > 0 else 0,
        'mean_uncertainty_incorrect': uncertainty_incorrect.mean() if len(uncertainty_incorrect) > 0 else 0,
    }
    
    return stats


def plot_uncertainty_statistics(stats, save_path=None):
    """
    Plot uncertainty statistics
    
    Args:
        stats: Statistics dictionary
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['mean_uncertainty', 'mean_uncertainty_correct', 'mean_uncertainty_incorrect']
    values = [stats[m] for m in metrics]
    labels = ['Overall', 'Correct Predictions', 'Incorrect Predictions']
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Uncertainty (Variance)', fontsize=12)
    ax.set_title('Uncertainty Analysis: Correct vs Incorrect Predictions', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Uncertainty statistics saved: {save_path}")
    
    return fig


if __name__ == "__main__":
    print("Testing uncertainty quantification module...")
    
    # This would normally load a trained model
    # For testing, we'll just verify imports work
    print(" All uncertainty functions imported successfully!")
    print("\nTo use:")
    print("1. Load trained model")
    print("2. Create MonteCarloDropout instance")
    print("3. Run predict_with_uncertainty()")
    print("4. Visualize results with visualize_uncertainty()")