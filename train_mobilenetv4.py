"""
MobileNetV4 training script for freshwater fish classification.
Implements two-stage training approach with proper optimization techniques.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
from torchvision import datasets
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import argparse
from datetime import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our custom transforms
from transforms import get_transforms, STAGE1_TRANSFORMS_CONFIG, STAGE2_TRANSFORMS_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fish class names (12 classes)
FISH_CLASSES = [
    'Black Rohu', 'Catla', 'Common Carp', 'Freshwater Shark', 
    'Grass Carp', 'Long-whiskered Catfish', 'Mirror Carp', 'Mrigal',
    'Nile Tilapia', 'Rohu', 'Silver Carp', 'Striped Catfish'
]

class FishDataset:
    """Dataset handler for freshwater fish classification."""
    
    def __init__(self, data_dir: str, transform_config: dict, val_split: float = 0.2):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the dataset directory
            transform_config: Configuration for transforms
            val_split: Validation split ratio
        """
        self.data_dir = Path(data_dir)
        self.transform_config = transform_config
        self.val_split = val_split
        self.transforms = get_transforms(transform_config)
        
        # Load dataset
        self.full_dataset = datasets.ImageFolder(
            root=str(self.data_dir),
            transform=self.transforms['train']
        )
        
        self.class_names = self.full_dataset.classes
        self.num_classes = len(self.class_names)
        
        # Split into train and validation
        val_size = int(len(self.full_dataset) * val_split)
        train_size = len(self.full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Update validation transforms
        self.val_dataset.dataset.transform = self.transforms['val']
        
        logger.info(f"Dataset loaded: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
        logger.info(f"Classes: {self.class_names}")
    
    def get_data_loaders(self, batch_size: int = 32, num_workers: int = 4):
        """Create data loaders for training and validation."""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader


class MobileNetV4Classifier:
    """MobileNetV4 model for fish classification."""
    
    def __init__(self, num_classes: int = 12, pretrained: bool = True):
        """
        Initialize MobileNetV4 model.
        
        Args:
            num_classes: Number of fish classes
            pretrained: Whether to use pretrained weights
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        
        # Create model using timm
        self.model = timm.create_model(
            'mobilenetv4_conv_medium',
            pretrained=pretrained,
            num_classes=num_classes
        )
        
        logger.info(f"MobileNetV4 model created with {num_classes} classes")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def freeze_backbone(self):
        """Freeze backbone parameters for stage 1 training."""
        for name, param in self.model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        logger.info("Backbone frozen - only classifier will be trained")
    
    def unfreeze_all(self):
        """Unfreeze all parameters for stage 2 training."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("All parameters unfrozen - full network will be fine-tuned")
    
    def get_model(self):
        """Get the model instance."""
        return self.model


class Trainer:
    """Training handler for MobileNetV4 fish classification."""
    
    def __init__(self, model, device, save_dir: str = 'checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            device: Training device (cuda/cpu)
            save_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = None
    
    def setup_optimizer(self, learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        """Setup AdamW optimizer."""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        logger.info(f"AdamW optimizer setup with lr={learning_rate}, weight_decay={weight_decay}")
    
    def setup_scheduler(self, T_max: int, eta_min: float = 1e-6):
        """Setup CosineAnnealingLR scheduler."""
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        logger.info(f"CosineAnnealingLR scheduler setup with T_max={T_max}, eta_min={eta_min}")
    
    def setup_loss_function(self, label_smoothing: float = 0.1):
        """Setup loss function with label smoothing."""
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"CrossEntropyLoss setup with label_smoothing={label_smoothing}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_acc = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs: int, stage_name: str = "Stage"):
        """Train the model for specified epochs."""
        logger.info(f"Starting {stage_name} training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\n{stage_name} Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Log results
            logger.info(f"{stage_name} Epoch {epoch+1}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                       f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = self.save_dir / f"best_model_{stage_name.lower()}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, self.best_model_path)
                logger.info(f"New best model saved with val_acc: {val_acc:.2f}%")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.save_dir / f"checkpoint_{stage_name.lower()}_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
        
        logger.info(f"{stage_name} training completed. Best val_acc: {self.best_val_acc:.2f}%")
        return self.history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train MobileNetV4 for fish classification')
    parser.add_argument('--data_dir', type=str, default='/home/nick/Desktop/INFO813 Project/processed/train',
                       help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs_stage1', type=int, default=10, help='Stage 1 epochs')
    parser.add_argument('--epochs_stage2', type=int, default=30, help='Stage 2 epochs')
    parser.add_argument('--lr_stage1', type=float, default=1e-3, help='Stage 1 learning rate')
    parser.add_argument('--lr_stage2', type=float, default=2e-5, help='Stage 2 learning rate')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Initialize dataset
    logger.info("Loading dataset...")
    dataset = FishDataset(
        data_dir=args.data_dir,
        transform_config=STAGE1_TRANSFORMS_CONFIG,
        val_split=0.2
    )
    
    train_loader, val_loader = dataset.get_data_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize model
    logger.info("Initializing MobileNetV4 model...")
    model_wrapper = MobileNetV4Classifier(num_classes=12, pretrained=True)
    model = model_wrapper.get_model()
    
    # Initialize trainer
    trainer = Trainer(model, device, save_dir=args.save_dir)
    
    # Stage 1: Freeze backbone, train classifier
    logger.info("\n" + "="*50)
    logger.info("STAGE 1: Training classifier only")
    logger.info("="*50)
    
    model_wrapper.freeze_backbone()
    trainer.setup_optimizer(learning_rate=args.lr_stage1)
    trainer.setup_scheduler(T_max=args.epochs_stage1)
    trainer.setup_loss_function(label_smoothing=0.1)
    
    # Update dataset transforms for stage 1
    dataset.transforms = get_transforms(STAGE1_TRANSFORMS_CONFIG)
    dataset.train_dataset.dataset.transform = dataset.transforms['train']
    
    history1 = trainer.train(train_loader, val_loader, args.epochs_stage1, "Stage1")
    
    # Stage 2: Unfreeze all, fine-tune entire network
    logger.info("\n" + "="*50)
    logger.info("STAGE 2: Fine-tuning entire network")
    logger.info("="*50)
    
    model_wrapper.unfreeze_all()
    trainer.setup_optimizer(learning_rate=args.lr_stage2)
    trainer.setup_scheduler(T_max=args.epochs_stage2)
    trainer.setup_loss_function(label_smoothing=0.05)
    
    # Update dataset transforms for stage 2
    dataset.transforms = get_transforms(STAGE2_TRANSFORMS_CONFIG)
    dataset.train_dataset.dataset.transform = dataset.transforms['train']
    
    history2 = trainer.train(train_loader, val_loader, args.epochs_stage2, "Stage2")
    
    # Save final model
    final_model_path = save_dir / "final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': dataset.class_names,
        'num_classes': dataset.num_classes,
        'history_stage1': history1,
        'history_stage2': history2,
        'best_val_acc': trainer.best_val_acc
    }, final_model_path)
    
    logger.info(f"Training completed! Final model saved to {final_model_path}")
    logger.info(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    
    # Save training history
    history_path = save_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump({
            'stage1': history1,
            'stage2': history2,
            'best_val_acc': trainer.best_val_acc
        }, f, indent=2)
    
    logger.info(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()