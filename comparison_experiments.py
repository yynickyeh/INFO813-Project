"""
Comparison Experiments Script for Fish Classification
PM7: Compare MobileNetV4 with MobileNetV3 and ConvNeXt-Tiny
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Torch metrics for evaluation
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Model libraries
import timm
from torchvision import datasets, transforms

# Fish class names
FISH_CLASSES = [
    'Black Rohu', 'Catla', 'Common Carp', 'Freshwater Shark', 
    'Grass Carp', 'Long-whiskered Catfish', 'Mirror Carp', 'Mrigal',
    'Nile Tilapia', 'Rohu', 'Silver Carp', 'Striped Catfish'
]

class ModelComparator:
    """
    Class for comparing different models on fish classification task.
    """
    
    def __init__(self, device='auto'):
        """
        Initialize model comparator.
        
        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else device)
        self.models = {}
        self.results = {}
        
        print(f"Using device: {self.device}")
    
    def create_model(self, model_name, num_classes=12, pretrained=True):
        """
        Create a model by name.
        
        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            
        Returns:
            Created model
        """
        if model_name == 'mobilenetv4':
            model = timm.create_model(
                'mobilenetv4_conv_medium',
                pretrained=pretrained,
                num_classes=num_classes
            )
        elif model_name == 'mobilenetv3':
            model = timm.create_model(
                'mobilenetv3_large_100',
                pretrained=pretrained,
                num_classes=num_classes
            )
        elif model_name == 'convnext_tiny':
            model = timm.create_model(
                'convnext_tiny',
                pretrained=pretrained,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model.to(self.device)
    
    def count_parameters(self, model):
        """
        Count model parameters.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }
    
    def measure_inference_time(self, model, input_tensor, num_runs=100):
        """
        Measure inference time.
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor
            num_runs: Number of inference runs
            
        Returns:
            Average inference time in milliseconds
        """
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)
        
        # Measure
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_tensor)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to ms
        
        return avg_time
    
    def evaluate_model(self, model, dataloader, model_name):
        """
        Evaluate a model on the dataset.
        
        Args:
            model: PyTorch model
            dataloader: Data loader
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                _, predictions = torch.max(outputs, 1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_outputs = np.array(all_outputs)
        
        # Calculate top-5 accuracy
        top5_correct = 0
        for i in range(len(all_targets)):
            top5_pred = np.argsort(all_outputs[i])[-5:]
            if all_targets[i] in top5_pred:
                top5_correct += 1
        top5_accuracy = top5_correct / len(all_targets)
        
        # Compute metrics
        results = {
            'accuracy': accuracy_score(all_targets, all_predictions),
            'top5_accuracy': top5_accuracy,
            'macro_f1': f1_score(all_targets, all_predictions, average='macro'),
            'macro_recall': recall_score(all_targets, all_predictions, average='macro'),
            'macro_precision': precision_score(all_targets, all_predictions, average='macro'),
            'weighted_f1': f1_score(all_targets, all_predictions, average='weighted'),
            'weighted_recall': recall_score(all_targets, all_predictions, average='weighted'),
            'weighted_precision': precision_score(all_targets, all_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(all_targets, all_predictions)
        }
        
        return results
    
    def run_comparison(self, data_dir, batch_size=32):
        """
        Run comparison experiments.
        
        Args:
            data_dir: Path to dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Comparison results
        """
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
        
        # Models to compare
        models_config = {
            'MobileNetV4': 'mobilenetv4',
            'MobileNetV3': 'mobilenetv3',
            'ConvNeXt-Tiny': 'convnext_tiny'
        }
        
        # Input tensor for inference time measurement
        sample_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        comparison_results = {}
        
        for display_name, model_name in models_config.items():
            print(f"\n{'='*60}")
            print(f"Evaluating {display_name}")
            print(f"{'='*60}")
            
            # Create model
            model = self.create_model(model_name)
            
            # Count parameters
            param_info = self.count_parameters(model)
            print(f"Total parameters: {param_info['total_params']:,}")
            print(f"Trainable parameters: {param_info['trainable_params']:,}")
            print(f"Model size: {param_info['model_size_mb']:.2f} MB")
            
            # Measure inference time
            inference_time = self.measure_inference_time(model, sample_input)
            print(f"Average inference time: {inference_time:.2f} ms")
            
            # Evaluate model
            eval_results = self.evaluate_model(model, dataloader, display_name)
            
            # Store results
            comparison_results[display_name] = {
                'model_name': model_name,
                'parameters': param_info,
                'inference_time_ms': inference_time,
                'metrics': eval_results
            }
            
            print(f"Accuracy: {eval_results['accuracy']:.4f}")
            print(f"Top-5 Accuracy: {eval_results['top5_accuracy']:.4f}")
            print(f"Macro F1-Score: {eval_results['macro_f1']:.4f}")
            
            # Clean up
            del model
            torch.cuda.empty_cache() if self.device.type == 'cuda' else None
        
        return comparison_results
    
    def create_comparison_table(self, results):
        """
        Create comparison table.
        
        Args:
            results: Comparison results
            
        Returns:
            DataFrame with comparison table
        """
        table_data = []
        
        for model_name, model_results in results.items():
            row = {
                'Model': model_name,
                'Parameters (M)': model_results['parameters']['total_params'] / 1e6,
                'Model Size (MB)': model_results['parameters']['model_size_mb'],
                'Inference Time (ms)': model_results['inference_time_ms'],
                'Accuracy': model_results['metrics']['accuracy'],
                'Top-5 Accuracy': model_results['metrics']['top5_accuracy'],
                'Macro F1-Score': model_results['metrics']['macro_f1'],
                'Macro Recall': model_results['metrics']['macro_recall'],
                'Macro Precision': model_results['metrics']['macro_precision'],
                'Weighted F1-Score': model_results['metrics']['weighted_f1'],
                'Weighted Recall': model_results['metrics']['weighted_recall'],
                'Weighted Precision': model_results['metrics']['weighted_precision']
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        return df
    
    def plot_comparison(self, results, save_dir):
        """
        Create comparison plots.
        
        Args:
            results: Comparison results
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create comparison table
        df = self.create_comparison_table(results)
        
        # Plot 1: Accuracy vs Model Size
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=df, x='Model Size (MB)', y='Accuracy', hue='Model', s=100)
        plt.title('Accuracy vs Model Size')
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Accuracy')
        
        # Plot 2: Accuracy vs Parameters
        plt.subplot(2, 2, 2)
        sns.scatterplot(data=df, x='Parameters (M)', y='Accuracy', hue='Model', s=100)
        plt.title('Accuracy vs Parameters')
        plt.xlabel('Parameters (M)')
        plt.ylabel('Accuracy')
        
        # Plot 3: Inference Time vs Accuracy
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='Inference Time (ms)', y='Accuracy', hue='Model', s=100)
        plt.title('Inference Time vs Accuracy')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Accuracy')
        
        # Plot 4: Model Size vs Inference Time
        plt.subplot(2, 2, 4)
        sns.scatterplot(data=df, x='Model Size (MB)', y='Inference Time (ms)', hue='Model', s=100)
        plt.title('Model Size vs Inference Time')
        plt.xlabel('Model Size (MB)')
        plt.ylabel('Inference Time (ms)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 5: Metrics comparison
        metrics_to_plot = ['Accuracy', 'Top-5 Accuracy', 'Macro F1-Score', 'Weighted F1-Score']
        
        plt.figure(figsize=(14, 8))
        df_melted = df.melt(id_vars=['Model'], value_vars=metrics_to_plot, 
                            var_name='Metric', value_name='Value')
        
        sns.barplot(data=df_melted, x='Model', y='Value', hue='Metric')
        plt.title('Performance Metrics Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 6: Model efficiency comparison
        plt.figure(figsize=(12, 6))
        
        # Calculate efficiency metrics
        df['Accuracy per MB'] = df['Accuracy'] / df['Model Size (MB)']
        df['Accuracy per M params'] = df['Accuracy'] / df['Parameters (M)']
        
        plt.subplot(1, 2, 1)
        sns.barplot(data=df, x='Model', y='Accuracy per MB')
        plt.title('Accuracy per Model Size (MB)')
        plt.ylabel('Accuracy / MB')
        plt.xticks(rotation=45)
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=df, x='Model', y='Accuracy per M params')
        plt.title('Accuracy per Million Parameters')
        plt.ylabel('Accuracy / M params')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'efficiency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results, save_dir):
        """
        Save comparison results.
        
        Args:
            results: Comparison results
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Create comparison table
        df = self.create_comparison_table(results)
        
        # Save table
        df.to_csv(os.path.join(save_dir, 'comparison_table.csv'), index=False)
        
        # Save detailed results
        with open(os.path.join(save_dir, 'detailed_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Create plots
        self.plot_comparison(results, save_dir)
        
        print(f"\nComparison results saved to {save_dir}")
        print(f"Comparison table: {os.path.join(save_dir, 'comparison_table.csv')}")
        print(f"Plots saved in: {save_dir}")


def main():
    """Main comparison function."""
    # Configuration
    data_dir = '/home/nick/Desktop/INFO813 Project/processed/val'
    save_dir = '/home/nick/Desktop/INFO813 Project/comparison_results'
    
    # Create comparator
    comparator = ModelComparator()
    
    # Run comparison
    print("Starting model comparison experiments...")
    results = comparator.run_comparison(data_dir)
    
    # Save results
    comparator.save_results(results, save_dir)
    
    print("\nComparison experiments completed!")


if __name__ == "__main__":
    main()