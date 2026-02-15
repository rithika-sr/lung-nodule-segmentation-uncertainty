"""
Training script for 3D U-Net lung nodule segmentation
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from models import get_model, count_parameters
from dataset import get_dataloaders
from utils import CombinedLoss, dice_coefficient, iou_score, save_checkpoint, plot_training_history


class Trainer:
    """
    Trainer class for U-Net segmentation model
    """
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader,
                 criterion,
                 optimizer,
                 device,
                 checkpoint_dir='../results/models/',
                 log_dir='../results/logs/'):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_dice_scores = []
        self.val_dice_scores = []
        self.best_val_dice = 0.0
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        """
        self.model.train()
        
        running_loss = 0.0
        running_dice = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        
        for batch_idx, (patches, masks, labels) in enumerate(pbar):
            # Move to device
            patches = patches.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(patches)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate metrics
            dice = dice_coefficient(outputs.detach(), masks.detach())
            
            # Update running metrics
            running_loss += loss.item()
            running_dice += dice
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}'
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_dice = running_dice / len(self.train_loader)
        
        return epoch_loss, epoch_dice
    
    def validate(self, epoch):
        """
        Validate the model
        """
        self.model.eval()
        
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]  ")
        
        with torch.no_grad():
            for patches, masks, labels in pbar:
                # Move to device
                patches = patches.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(patches)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                
                # Calculate metrics
                dice = dice_coefficient(outputs, masks)
                iou = iou_score(outputs, masks)
                
                # Update running metrics
                running_loss += loss.item()
                running_dice += dice
                running_iou += iou
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice': f'{dice:.4f}',
                    'iou': f'{iou:.4f}'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        epoch_dice = running_dice / len(self.val_loader)
        epoch_iou = running_iou / len(self.val_loader)
        
        return epoch_loss, epoch_dice, epoch_iou
    
    def train(self, num_epochs):
        """
        Main training loop
        """
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Number of epochs: {num_epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_dice = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_dice, val_iou = self.validate(epoch)
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_dice_scores.append(train_dice)
            self.val_dice_scores.append(val_dice)
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Dice/train', train_dice, epoch)
            self.writer.add_scalar('Dice/val', val_dice, epoch)
            self.writer.add_scalar('IoU/val', val_iou, epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
            print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f} | Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                best_model_path = self.checkpoint_dir / 'best_model.pth'
                save_checkpoint(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    val_loss, 
                    best_model_path
                )
                print(f"  🎯 New best model! Val Dice: {val_dice:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    checkpoint_path
                )
            
            print("-" * 70)
        
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"{'='*70}")
        print(f"Best validation Dice score: {self.best_val_dice:.4f}")
        
        # Save training history plot
        history_plot_path = Path('../results/plots/training_history.png')
        history_plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_training_history(
            self.train_losses,
            self.val_losses,
            self.train_dice_scores,
            self.val_dice_scores,
            save_path=str(history_plot_path)
        )
        
        self.writer.close()


def main(args):
    """
    Main training function
    """
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print("\nCreating model...")
    model = get_model(
        in_channels=1,
        out_channels=1,
        init_features=args.init_features,
        dropout_rate=args.dropout_rate
    )
    model = model.to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Train
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train 3D U-Net for lung nodule segmentation')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='../data/processed/',
                        help='Directory with preprocessed data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--init_features', type=int, default=16,
                        help='Initial number of features in U-Net')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout rate for uncertainty quantification')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Save parameters
    parser.add_argument('--checkpoint_dir', type=str, default='../results/models/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='../results/logs/',
                        help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    main(args)