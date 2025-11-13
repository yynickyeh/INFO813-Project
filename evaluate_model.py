"""
Model Evaluation Script for MobileNetV4 Fish Classification
PM6: Model Evaluation with accuracy, macro-F1, recall, confusion matrix, and Grad-CAM visualization
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import yaml
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Metrics for evaluation
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report

# Grad-CAM implementation
import timm
import cv2
from typing import List, Callable, Optional, Tuple
# #任务PM6：模型评估 工具/方法：- 指标：accuracy、macro-F1、recall、confusion matrix- 使用 torchmetrics- 绘制 Grad-CAM 热力图 交付成果：eval_results.csv、plots/confusion_matrix.png、plots/gradcam_samples.png
# Fish class names
FISH_CLASSES = [
    'Black Rohu', 'Catla', 'Common Carp', 'Freshwater Shark', 
    'Grass Carp', 'Long-whiskered Catfish', 'Mirror Carp', 'Mrigal',
    'Nile Tilapia', 'Rohu', 'Silver Carp', 'Striped Catfish'
]

class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[str], use_cuda: bool = True):
        """
        Initialize Grad-CAM.
        
        Args:
            model: PyTorch model
            target_layers: List of layer names to hook
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.target_layers = target_layers
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = model.cuda()
        
        # Hook storage
        self.gradients = {}
        self.activations = {}
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for target layers."""
        def forward_hook(module, input, output, name):
            self.activations[name] = output
        
        def backward_hook(module, grad_input, grad_output, name):
            self.gradients[name] = grad_output[0]
        
        # Find and register hooks for target layers
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layers):
                module.register_forward_hook(lambda m, i, o, n=name: forward_hook(m, i, o, n))
                module.register_backward_hook(lambda m, gi, go, n=name: backward_hook(m, gi, go, n))
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Generate Grad-CAM for a specific class.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            class_idx: Target class index
            
        Returns:
            CAM heatmap as numpy array
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        class_score = output[0, class_idx]
        class_score.backward()
        
        # Get gradients and activations
        target_layer = self.target_layers[0]  # Use first target layer
        gradients = self.gradients[target_layer]
        activations = self.activations[target_layer]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.squeeze().detach().cpu().numpy()
    
    def visualize_cam(self, image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            image: Original image (H, W, 3)
            cam: CAM heatmap (H, W)
            alpha: Transparency factor
            
        Returns:
            Overlayed image
        """
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
        
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        
        # Overlay on original image
        overlay = image * (1 - alpha) + heatmap * alpha
        
        return np.uint8(overlay)


class ModelEvaluator:
    """
    Comprehensive model evaluator for fish classification.
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'auto'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize model
        self.model = self._load_model()
        
        # Initialize metrics
        self._init_metrics()
        
        # Results storage
        self.results = {}
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_model(self) -> nn.Module:
        """Load trained model."""
        # Create model
        model = timm.create_model(
            'mobilenetv4_conv_medium',
            pretrained=False,
            num_classes=12
        )
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {self.model_path}")
        return model
    
    def _init_metrics(self):
        """Initialize evaluation metrics."""
        # No initialization needed for sklearn metrics
        pass
    
    def _reset_metrics(self):
        """Reset all metrics."""
        # No reset needed for sklearn metrics
        pass
    
    def evaluate_dataset(self, data_dir: str, batch_size: int = 32) -> dict:
        """
        Evaluate model on entire dataset.
        
        Args:
            data_dir: Path to dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        from torchvision import datasets, transforms
        
        # Create dataset and dataloader
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=4
        )
        
        # Evaluation loop
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_outputs = np.array(all_outputs)
        
        # Calculate top-5 accuracy
        top5_correct = 0
        for i in range(len(all_targets)):
            top5_pred = np.argsort(all_outputs[i])[-5:]
            if all_targets[i] in top5_pred:
                top5_correct += 1
        top5_accuracy = top5_correct / len(all_targets)
        
        # Compute metrics using sklearn
        results = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'top5_accuracy': top5_accuracy,
            'macro_f1': f1_score(all_targets, all_predictions, average='macro'),
            'macro_recall': recall_score(all_targets, all_predictions, average='macro'),
            'macro_precision': precision_score(all_targets, all_predictions, average='macro'),
            'weighted_f1': f1_score(all_targets, all_predictions, average='weighted'),
            'weighted_recall': recall_score(all_targets, all_predictions, average='weighted'),
            'weighted_precision': precision_score(all_targets, all_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(all_targets, all_predictions),
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probabilities.tolist(),
            'class_names': dataset.classes
        }
        
        return results
    
    def generate_confusion_matrix_plot(self, confusion_matrix: np.ndarray, class_names: List[str], 
                                    save_path: str = None) -> plt.Figure:
        """
        Generate and save confusion matrix plot.
        
        Args:
            confusion_matrix: Confusion matrix array
            class_names: List of class names
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        return plt.gcf()
    
    def generate_gradcam_samples(self, data_dir: str, num_samples: int = 5, 
                               save_dir: str = None) -> List[np.ndarray]:
        """
        Generate Grad-CAM visualizations for sample images.
        
        Args:
            data_dir: Path to dataset
            num_samples: Number of samples per class
            save_dir: Directory to save visualizations
            
        Returns:
            List of Grad-CAM visualizations
        """
        from torchvision import datasets, transforms
        
        # Create dataset
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        
        # Initialize Grad-CAM
        grad_cam = GradCAM(self.model, target_layers=['blocks.3.10.pw_exp.conv'])
        
        # Create save directory
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        gradcam_images = []
        
        # Generate samples for each class
        for class_idx, class_name in enumerate(dataset.classes):
            # Find samples for this class
            class_samples = [i for i, (_, label) in enumerate(dataset.samples) if label == class_idx]
            
            # Select random samples
            if len(class_samples) > num_samples:
                selected_samples = np.random.choice(class_samples, num_samples, replace=False)
            else:
                selected_samples = class_samples
            
            # Generate Grad-CAM for each sample
            for sample_idx in selected_samples:
                # Load and preprocess image
                image_path, _ = dataset.samples[sample_idx]
                original_image = Image.open(image_path).convert('RGB')
                original_array = np.array(original_image.resize((224, 224)))
                
                # Prepare input tensor
                input_tensor = transform(original_image).unsqueeze(0).to(self.device)
                
                # Generate Grad-CAM
                cam = grad_cam.generate_cam(input_tensor, class_idx)
                cam_image = grad_cam.visualize_cam(original_array, cam)
                
                gradcam_images.append(cam_image)
                
                # Save visualization
                if save_dir:
                    save_path = os.path.join(save_dir, f"{class_name}_{sample_idx}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
        
        if save_dir:
            print(f"Grad-CAM samples saved to {save_dir}")
        
        return gradcam_images
    
    def generate_class_report(self, results: dict, save_path: str = None) -> pd.DataFrame:
        """
        Generate per-class performance report.
        
        Args:
            results: Evaluation results dictionary
            save_path: Path to save the report
            
        Returns:
            DataFrame with per-class metrics
        """
        
        # Generate classification report
        report = classification_report(
            results['targets'], 
            results['predictions'],
            target_names=results['class_names'],
            output_dict=True
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(report).transpose()
        
        # Save report
        if save_path:
            df.to_csv(save_path)
            print(f"Class report saved to {save_path}")
        
        return df
    
    def save_evaluation_results(self, results: dict, save_dir: str):
        """
        Save all evaluation results.
        
        Args:
            results: Evaluation results dictionary
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
        
        # Save metrics summary
        metrics_summary = {
            'accuracy': results['accuracy'],
            'top5_accuracy': results['top5_accuracy'],
            'macro_f1': results['macro_f1'],
            'macro_recall': results['macro_recall'],
            'macro_precision': results['macro_precision'],
            'weighted_f1': results['weighted_f1'],
            'weighted_recall': results['weighted_recall'],
            'weighted_precision': results['weighted_precision']
        }
        
        with open(os.path.join(save_dir, 'eval_results.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Save confusion matrix plot
        self.generate_confusion_matrix_plot(
            results['confusion_matrix'],
            results['class_names'],
            os.path.join(save_dir, 'plots', 'confusion_matrix.png')
        )
        
        # Generate Grad-CAM samples
        self.generate_gradcam_samples(
            '/home/nick/Desktop/INFO813 Project/processed/val',
            num_samples=3,
            save_dir=os.path.join(save_dir, 'plots', 'gradcam_samples')
        )
        
        # Generate class report
        class_report = self.generate_class_report(
            results,
            os.path.join(save_dir, 'eval_results.csv')
        )
        
        print(f"Evaluation results saved to {save_dir}")
        return metrics_summary


def main():
    """Main evaluation function."""
    # Configuration
    model_path = '/home/nick/Desktop/INFO813 Project/checkpoints/best_model_stage2.pth'
    config_path = '/home/nick/Desktop/INFO813 Project/train_config.yaml'
    data_dir = '/home/nick/Desktop/INFO813 Project/processed/val'
    save_dir = '/home/nick/Desktop/INFO813 Project/evaluation_results'
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path, config_path)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate_dataset(data_dir)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.4f}")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Weighted F1-Score: {results['weighted_f1']:.4f}")
    print(f"Weighted Recall: {results['weighted_recall']:.4f}")
    print(f"Weighted Precision: {results['weighted_precision']:.4f}")
    
    # Save results
    evaluator.save_evaluation_results(results, save_dir)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()