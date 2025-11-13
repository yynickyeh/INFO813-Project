import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
from PIL import Image
import random
import math
#任务4：可视化 工具/方法：- 类别分布图（条形+饼图）- 每类 3 张示例图网格- 图像尺寸分布直方图 交付成果：plots/class_distribution.png、plots/sample_examples.png
def create_visualizations():
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Ensure plots directory exists
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    print("Creating Comprehensive Visualizations (DA3)")
    print("=" * 60)
    
    # Read class statistics
    df = pd.read_csv('class_counts.csv')
    
    # 1. Create sample examples grid (3 samples per class)
    print("\n1. Creating sample examples grid...")
    
    # Get all categories
    categories = df['Category'].tolist()
    
    # Create a figure with subplots for each category
    fig, axes = plt.subplots(len(categories), 3, figsize=(15, 4*len(categories)))
    fig.suptitle('Fish Dataset - Sample Examples (3 per class)', fontsize=16, fontweight='bold')
    
    for i, category in enumerate(categories):
        # Get train directory for this category
        train_dir = Path('processed/train') / category
        
        if train_dir.exists():
            # Get all image files
            image_files = [f for f in train_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            # Randomly select 3 images (or fewer if not enough)
            selected_files = random.sample(image_files, min(3, len(image_files)))
            
            for j, img_file in enumerate(selected_files):
                try:
                    # Load and display image
                    img = Image.open(img_file)
                    
                    ax = axes[i, j] if len(categories) > 1 else axes[j]
                    ax.imshow(img)
                    ax.axis('off')
                    
                    # Add title only for top row
                    if j == 0:
                        ax.set_title(f'{category}', fontsize=12, fontweight='bold', pad=10)
                    
                    # Add filename as subtitle
                    ax.set_xlabel(img_file.name, fontsize=8, wrap=True)
                    
                except Exception as e:
                    print(f"Error loading {img_file}: {e}")
                    # Create empty subplot if image can't be loaded
                    ax = axes[i, j] if len(categories) > 1 else axes[j]
                    ax.text(0.5, 0.5, f'Error loading\n{img_file.name}', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.axis('off')
        else:
            # Create empty subplots if directory doesn't exist
            for j in range(3):
                ax = axes[i, j] if len(categories) > 1 else axes[j]
                ax.text(0.5, 0.5, f'No images\nfor {category}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('plots/sample_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✓ Sample examples grid saved to plots/sample_examples.png")
    
    # 2. Analyze image dimensions
    print("\n2. Analyzing image dimensions...")
    
    image_dimensions = []
    aspect_ratios = []
    file_sizes = []
    
    # Sample images from each category for analysis
    sample_size = min(50, max(5, len(categories) * 4))  # Sample at least 5 images per category on average
    
    for category in categories:
        train_dir = Path('processed/train') / category
        if train_dir.exists():
            image_files = [f for f in train_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            # Sample images from this category
            sample_files = random.sample(image_files, min(len(image_files), max(5, sample_size // len(categories))))
            
            for img_file in sample_files:
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                        image_dimensions.append((width, height))
                        aspect_ratios.append(width / height)
                        file_sizes.append(img_file.stat().st_size / 1024)  # Size in KB
                except Exception as e:
                    print(f"   Warning: Could not analyze {img_file}: {e}")
    
    if image_dimensions:
        # Convert to numpy arrays for analysis
        widths = np.array([dim[0] for dim in image_dimensions])
        heights = np.array([dim[1] for dim in image_dimensions])
        aspect_ratios = np.array(aspect_ratios)
        file_sizes = np.array(file_sizes)
        
        print(f"   Analyzed {len(image_dimensions)} images")
        print(f"   Width range: {widths.min()} - {widths.max()} pixels (mean: {widths.mean():.1f})")
        print(f"   Height range: {heights.min()} - {heights.max()} pixels (mean: {heights.mean():.1f})")
        print(f"   Aspect ratio range: {aspect_ratios.min():.2f} - {aspect_ratios.max():.2f} (mean: {aspect_ratios.mean():.2f})")
        print(f"   File size range: {file_sizes.min():.1f} - {file_sizes.max():.1f} KB (mean: {file_sizes.mean():.1f} KB)")
        
        # 3. Create image size distribution histograms
        print("\n3. Creating image size distribution histograms...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Image Size Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Width distribution
        axes[0, 0].hist(widths, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Image Width Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(widths.mean(), color='red', linestyle='--', label=f'Mean: {widths.mean():.1f}')
        axes[0, 0].legend()
        
        # Height distribution
        axes[0, 1].hist(heights, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Image Height Distribution', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(heights.mean(), color='red', linestyle='--', label=f'Mean: {heights.mean():.1f}')
        axes[0, 1].legend()
        
        # Aspect ratio distribution
        axes[1, 0].hist(aspect_ratios, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Aspect Ratio (Width/Height)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(aspect_ratios.mean(), color='red', linestyle='--', label=f'Mean: {aspect_ratios.mean():.2f}')
        axes[1, 0].legend()
        
        # File size distribution
        axes[1, 1].hist(file_sizes, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('File Size Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('File Size (KB)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(file_sizes.mean(), color='red', linestyle='--', label=f'Mean: {file_sizes.mean():.1f} KB')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('plots/image_size_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Image size distribution histograms saved to plots/image_size_distribution.png")
        
        # 4. Create scatter plot of width vs height
        plt.figure(figsize=(10, 8))
        plt.scatter(widths, heights, alpha=0.6, s=30, c=aspect_ratios, cmap='viridis')
        plt.colorbar(label='Aspect Ratio')
        plt.xlabel('Width (pixels)', fontsize=12)
        plt.ylabel('Height (pixels)', fontsize=12)
        plt.title('Image Dimensions Scatter Plot', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add reference lines for common aspect ratios
        max_dim = max(widths.max(), heights.max())
        plt.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='1:1 (Square)')
        plt.plot([0, max_dim], [0, max_dim*4/3], 'g--', alpha=0.5, label='4:3')
        plt.plot([0, max_dim], [0, max_dim*16/9], 'b--', alpha=0.5, label='16:9')
        plt.plot([0, max_dim*3/4], [0, max_dim], 'orange', linestyle='--', alpha=0.5, label='3:4')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/dimensions_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Dimensions scatter plot saved to plots/dimensions_scatter.png")
        
        # 5. Create comprehensive visualization summary
        print("\n4. Creating visualization summary...")
        
        # Create a summary figure with key statistics
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fish Dataset - Comprehensive Visualization Summary', fontsize=16, fontweight='bold')
        
        # Class distribution (recreate from previous analysis)
        df_sorted = df.sort_values('Total', ascending=True)
        axes[0, 0].barh(df_sorted['Category'], df_sorted['Total'], color=sns.color_palette("husl", len(df_sorted)))
        axes[0, 0].set_title('Class Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Images')
        
        # Train/Val/Test split
        categories_short = [cat.replace(' ', '\n') for cat in df['Category']]
        x = np.arange(len(categories_short))
        width = 0.25
        
        axes[0, 1].bar(x - width, df['train'], width, label='Train', color='#2ecc71')
        axes[0, 1].bar(x, df['val'], width, label='Validation', color='#3498db')
        axes[0, 1].bar(x + width, df['test'], width, label='Test', color='#e74c3c')
        axes[0, 1].set_title('Dataset Split by Category', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(categories_short, rotation=45, ha='right', fontsize=8)
        axes[0, 1].legend()
        
        # Class balance
        axes[0, 2].bar(df['Category'], df['Class Balance'], color=sns.color_palette("Reds", len(df)))
        axes[0, 2].set_title('Class Balance (relative to max)', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('Balance Ratio')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Image dimensions summary (only first 3 to fit in 2x3 grid)
        dimension_data = [widths, heights, aspect_ratios]
        dimension_labels = ['Width', 'Height', 'Aspect Ratio']
        colors = ['skyblue', 'lightgreen', 'orange']
        
        for i, (data, label, color) in enumerate(zip(dimension_data, dimension_labels, colors)):
            row, col = 1, i
            axes[row, col].hist(data, bins=20, alpha=0.7, color=color, edgecolor='black')
            axes[row, col].set_title(f'{label} Distribution', fontsize=12, fontweight='bold')
            axes[row, col].set_xlabel(label)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].axvline(data.mean(), color='red', linestyle='--',
                                  label=f'Mean: {data.mean():.1f}' if label != 'Aspect Ratio' else f'Mean: {data.mean():.2f}')
            axes[row, col].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Comprehensive summary saved to plots/comprehensive_summary.png")
    
    print(f"\n✅ All visualizations created successfully!")
    print(f"📊 Generated files in plots/ directory:")
    print(f"   - sample_examples.png (3 sample images per class)")
    print(f"   - image_size_distribution.png (width, height, aspect ratio, file size histograms)")
    print(f"   - dimensions_scatter.png (width vs height scatter plot)")
    print(f"   - comprehensive_summary.png (overview of all key visualizations)")
    
    # Save image analysis statistics to a text file
    if image_dimensions:
        with open('plots/image_analysis_stats.txt', 'w') as f:
            f.write("Image Analysis Statistics\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total images analyzed: {len(image_dimensions)}\n\n")
            f.write("Width Statistics:\n")
            f.write(f"  Min: {widths.min()} pixels\n")
            f.write(f"  Max: {widths.max()} pixels\n")
            f.write(f"  Mean: {widths.mean():.1f} pixels\n")
            f.write(f"  Std: {widths.std():.1f} pixels\n\n")
            f.write("Height Statistics:\n")
            f.write(f"  Min: {heights.min()} pixels\n")
            f.write(f"  Max: {heights.max()} pixels\n")
            f.write(f"  Mean: {heights.mean():.1f} pixels\n")
            f.write(f"  Std: {heights.std():.1f} pixels\n\n")
            f.write("Aspect Ratio Statistics:\n")
            f.write(f"  Min: {aspect_ratios.min():.2f}\n")
            f.write(f"  Max: {aspect_ratios.max():.2f}\n")
            f.write(f"  Mean: {aspect_ratios.mean():.2f}\n")
            f.write(f"  Std: {aspect_ratios.std():.2f}\n\n")
            f.write("File Size Statistics:\n")
            f.write(f"  Min: {file_sizes.min():.1f} KB\n")
            f.write(f"  Max: {file_sizes.max():.1f} KB\n")
            f.write(f"  Mean: {file_sizes.mean():.1f} KB\n")
            f.write(f"  Std: {file_sizes.std():.1f} KB\n")
        
        print(f"📄 Image analysis statistics saved to: plots/image_analysis_stats.txt")

if __name__ == "__main__":
    create_visualizations()