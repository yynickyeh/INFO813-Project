"""
Data augmentation strategies for freshwater fish classification.
This module provides training and validation transforms optimized for fish images.
"""

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import random
from typing import Dict, Any


class FishAugmentation:
    """
    Data augmentation pipeline specifically designed for freshwater fish images.
    Includes augmentations that preserve fish characteristics while improving model robustness.
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 normalize_mean: tuple = (0.485, 0.456, 0.406),
                 normalize_std: tuple = (0.229, 0.224, 0.225),
                 augmentation_strength: str = 'medium'):
        """
        Initialize augmentation pipeline.
        
        Args:
            image_size: Target image size for resizing
            normalize_mean: Mean values for normalization (ImageNet standards)
            normalize_std: Standard deviation values for normalization (ImageNet standards)
            augmentation_strength: 'light', 'medium', or 'heavy' augmentation
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.augmentation_strength = augmentation_strength
        
    def get_train_transforms(self) -> T.Compose:
        """
        Get training transforms with data augmentation.
        
        Returns:
            Composed transform pipeline for training
        """
        if self.augmentation_strength == 'light':
            return self._get_light_augmentation()
        elif self.augmentation_strength == 'medium':
            return self._get_medium_augmentation()
        elif self.augmentation_strength == 'heavy':
            return self._get_heavy_augmentation()
        else:
            return self._get_medium_augmentation()
    
    def get_val_transforms(self) -> T.Compose:
        """
        Get validation transforms (no augmentation, only preprocessing).
        
        Returns:
            Composed transform pipeline for validation
        """
        return T.Compose([
            T.Resize((self.image_size, self.image_size), 
                    interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def get_test_transforms(self) -> T.Compose:
        """
        Get test transforms (same as validation).
        
        Returns:
            Composed transform pipeline for testing
        """
        return self.get_val_transforms()
    
    def _get_light_augmentation(self) -> T.Compose:
        """Light augmentation for initial training stages."""
        return T.Compose([
            T.Resize((self.image_size + 32, self.image_size + 32), 
                    interpolation=InterpolationMode.BILINEAR),
            T.RandomResizedCrop(self.image_size, scale=(0.9, 1.0), 
                               ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.ToTensor(),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def _get_medium_augmentation(self) -> T.Compose:
        """Medium augmentation for main training."""
        return T.Compose([
            T.Resize((self.image_size + 32, self.image_size + 32), 
                    interpolation=InterpolationMode.BILINEAR),
            T.RandomResizedCrop(self.image_size, scale=(0.8, 1.0), 
                               ratio=(0.8, 1.2)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomRotation(degrees=10),
            T.ToTensor(),
            T.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), 
                           value='random'),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])
    
    def _get_heavy_augmentation(self) -> T.Compose:
        """Heavy augmentation for fine-tuning and robustness."""
        return T.Compose([
            T.Resize((self.image_size + 64, self.image_size + 64), 
                    interpolation=InterpolationMode.BILINEAR),
            T.RandomResizedCrop(self.image_size, scale=(0.7, 1.0), 
                               ratio=(0.7, 1.3)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
            T.ToTensor(),
            T.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3), 
                           value='random'),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])


def get_transforms(config: Dict[str, Any]) -> Dict[str, T.Compose]:
    """
    Factory function to get transforms based on configuration.
    
    Args:
        config: Configuration dictionary containing transform parameters
        
    Returns:
        Dictionary with 'train', 'val', and 'test' transforms
    """
    augmentation = FishAugmentation(
        image_size=config.get('image_size', 224),
        normalize_mean=config.get('normalize_mean', (0.485, 0.456, 0.406)),
        normalize_std=config.get('normalize_std', (0.229, 0.224, 0.225)),
        augmentation_strength=config.get('augmentation_strength', 'medium')
    )
    
    return {
        'train': augmentation.get_train_transforms(),
        'val': augmentation.get_val_transforms(),
        'test': augmentation.get_test_transforms()
    }


# Default configurations for different training stages
STAGE1_TRANSFORMS_CONFIG = {
    'image_size': 224,
    'normalize_mean': (0.485, 0.456, 0.406),
    'normalize_std': (0.229, 0.224, 0.225),
    'augmentation_strength': 'light'  # Light augmentation for stage 1
}

STAGE2_TRANSFORMS_CONFIG = {
    'image_size': 224,
    'normalize_mean': (0.485, 0.456, 0.406),
    'normalize_std': (0.229, 0.224, 0.225),
    'augmentation_strength': 'medium'  # Medium augmentation for stage 2
}

FINETUNE_TRANSFORMS_CONFIG = {
    'image_size': 224,
    'normalize_mean': (0.485, 0.456, 0.406),
    'normalize_std': (0.229, 0.224, 0.225),
    'augmentation_strength': 'heavy'  # Heavy augmentation for fine-tuning
}


if __name__ == "__main__":
    # Example usage and testing
    config = STAGE2_TRANSFORMS_CONFIG
    transforms_dict = get_transforms(config)
    
    print("Transform configurations:")
    print(f"Train transforms: {transforms_dict['train']}")
    print(f"Val transforms: {transforms_dict['val']}")
    print(f"Test transforms: {transforms_dict['test']}")
    
    # Test with a sample tensor
    sample_tensor = torch.randn(3, 256, 256)
    print(f"\nSample tensor shape: {sample_tensor.shape}")
    
    # Note: To test with actual images, you would need PIL Image
    # from PIL import Image
    # image = Image.open("path/to/fish/image.jpg")
    # transformed = transforms_dict['train'](image)
    # print(f"Transformed image shape: {transformed.shape}")