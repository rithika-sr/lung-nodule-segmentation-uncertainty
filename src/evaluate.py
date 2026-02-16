"""
Evaluate trained model with uncertainty quantification
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from models import get_model
from dataset import get_dataloaders
from uncertainty import MonteCarloDropout, visualize_uncertainty, analyze_uncertainty_statistics, plot_uncertainty_statistics
from utils import dice_coefficient, iou_score


def evaluate_model(model_path, data_dir, n_mc_samples=20, num_visualizations=5):
    """
    Evaluate model with uncertainty quantification
    
    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to preprocessed data
        n_mc_samples: Number of Monte Carlo samples
        num_visualizations: Number of samples to visualize
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = get_model(in_channels=1, out_channels=1, init_features=16, dropout_rate=0.2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f" Model loaded from: {model_path}")
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=1, num_workers=0)
    print(f" Test samples: {len(test_loader.dataset)}")
    
    # Create Monte Carlo Dropout evaluator
    print(f"\nCreating Monte Carlo Dropout evaluator with {n_mc_samples} samples...")
    mc_dropout = MonteCarloDropout(model, device, n_samples=n_mc_samples)
    
    # Evaluate
    print("\nEvaluating model with uncertainty quantification...")
    results = mc_dropout.evaluate_uncertainty(test_loader)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    predictions = results['predictions']
    masks = results['masks']
    uncertainties = results['uncertainties']
    
    # Overall metrics
    total_dice = 0
    total_iou = 0
    
    for i in range(len(predictions)):
        dice = dice_coefficient(predictions[i], masks[i])
        iou = iou_score(predictions[i], masks[i])
        total_dice += dice
        total_iou += iou
    
    avg_dice = total_dice / len(predictions)
    avg_iou = total_iou / len(predictions)
    
    print(f"\n Average Dice Score: {avg_dice:.4f}")
    print(f" Average IoU Score: {avg_iou:.4f}")
    
    # Uncertainty statistics
    print(f"\n{'='*70}")
    print("UNCERTAINTY ANALYSIS")
    print(f"{'='*70}")
    
    stats = analyze_uncertainty_statistics(uncertainties, predictions, masks)
    
    print(f"\n Overall Uncertainty Statistics:")
    print(f"   Mean uncertainty: {stats['mean_uncertainty']:.6f}")
    print(f"   Std uncertainty: {stats['std_uncertainty']:.6f}")
    print(f"   Max uncertainty: {stats['max_uncertainty']:.6f}")
    print(f"   Min uncertainty: {stats['min_uncertainty']:.6f}")
    
    print(f"\n Uncertainty by Correctness:")
    print(f"   Correct predictions: {stats['mean_uncertainty_correct']:.6f}")
    print(f"   Incorrect predictions: {stats['mean_uncertainty_incorrect']:.6f}")
    
    if stats['mean_uncertainty_incorrect'] > stats['mean_uncertainty_correct']:
        print(f"\n Good! Model is MORE uncertain on incorrect predictions!")
        print(f"   Ratio: {stats['mean_uncertainty_incorrect'] / stats['mean_uncertainty_correct']:.2f}x")
    else:
        print(f"\n  Model uncertainty needs calibration")
    
    # Save uncertainty statistics plot
    plots_dir = Path('../results/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_uncertainty_statistics(
        stats,
        save_path=str(plots_dir / 'uncertainty_statistics.png')
    )
    
    # Visualize samples
    print(f"\n{'='*70}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    images = results['images']
    
    # Select diverse samples to visualize
    num_samples = min(num_visualizations, len(images))
    indices = np.linspace(0, len(images)-1, num_samples, dtype=int)
    
    for idx in indices:
        print(f"\nVisualizing sample {idx+1}/{num_samples}...")
        
        # Find best slice (one with most mask content)
        mask_sums = masks[idx].squeeze().sum(dim=(1, 2))
        best_slice = mask_sums.argmax().item()
        
        save_path = plots_dir / f'uncertainty_sample_{idx}.png'
        visualize_uncertainty(
            images[idx],
            masks[idx],
            predictions[idx],
            uncertainties[idx],
            slice_idx=best_slice,
            save_path=str(save_path)
        )
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\n Results saved to: {plots_dir}")
    print(f"\n Key Findings:")
    print(f"   • Model achieved {avg_dice:.2%} Dice score")
    print(f"   • Uncertainty is {'higher' if stats['mean_uncertainty_incorrect'] > stats['mean_uncertainty_correct'] else 'lower'} on incorrect predictions")
    print(f"   • Generated {num_samples} uncertainty visualizations")
    
    return results, stats


def main(args):
    """
    Main evaluation function
    """
    evaluate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        n_mc_samples=args.n_mc_samples,
        num_visualizations=args.num_visualizations
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model with uncertainty quantification')
    
    parser.add_argument('--model_path', type=str, 
                        default='../results/models/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default='../data/processed/',
                        help='Path to preprocessed data')
    parser.add_argument('--n_mc_samples', type=int, 
                        default=20,
                        help='Number of Monte Carlo samples for uncertainty')
    parser.add_argument('--num_visualizations', type=int, 
                        default=5,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    main(args)