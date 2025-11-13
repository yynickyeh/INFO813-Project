import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns
#任务2：样本统计 工具/方法：统计各类样本数量、比例 交付成果：柱状图
def generate_sample_statistics():
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Read the existing class counts
    df = pd.read_csv('class_counts.csv')
    
    # Create plots directory if it doesn't exist
    plots_dir = Path('plots')
    plots_dir.mkdir(exist_ok=True)
    
    print("Generating Sample Statistics (DA2)")
    print("=" * 50)
    
    # 1. Enhanced class distribution analysis
    print("\n1. Class Distribution Analysis:")
    
    # Calculate additional statistics
    df['Percentage'] = (df['Total'] / df['Total'].sum()) * 100
    df['Class Balance'] = df['Total'] / df['Total'].max()
    
    # Sort by total count
    df_sorted = df.sort_values('Total', ascending=True)
    
    print(f"Total classes: {len(df)}")
    print(f"Total images: {df['Total'].sum()}")
    print(f"Images per class - Min: {df['Total'].min()}, Max: {df['Total'].max()}, Mean: {df['Total'].mean():.1f}")
    print(f"Class balance ratio: {df['Total'].min()}/{df['Total'].max()} = {df['Total'].min()/df['Total'].max():.3f}")
    
    # 2. Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(df_sorted['Category'], df_sorted['Total'], color=sns.color_palette("husl", len(df_sorted)))
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                f'{int(width)} ({df_sorted.iloc[i]["Percentage"]:.1f}%)', 
                ha='left', va='center', fontsize=10)
    
    plt.title('Fish Dataset - Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Number of Images', fontsize=12)
    plt.ylabel('Fish Categories', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Create pie chart
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", len(df))
    wedges, texts, autotexts = plt.pie(df['Total'], labels=df['Category'], autopct='%1.1f%%', 
                                      colors=colors, startangle=90, textprops={'fontsize': 10})
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.title('Fish Dataset - Class Distribution (Pie Chart)', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('plots/class_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Create train/val/test split visualization
    plt.figure(figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    categories = df['Category']
    train_counts = df['train']
    val_counts = df['val']
    test_counts = df['test']
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    bars1 = ax.bar(x - width, train_counts, width, label='Train', color='#2ecc71')
    bars2 = ax.bar(x, val_counts, width, label='Validation', color='#3498db')
    bars3 = ax.bar(x + width, test_counts, width, label='Test', color='#e74c3c')
    
    ax.set_xlabel('Fish Categories', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Dataset Split Distribution by Category', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height)}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('plots/split_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Create enhanced CSV with additional statistics
    enhanced_stats = df.copy()
    
    # Add statistical measures
    enhanced_stats['Std_Dev'] = (enhanced_stats[['train', 'val', 'test']].std(axis=1)).round(1)
    enhanced_stats['CV'] = ((enhanced_stats[['train', 'val', 'test']].std(axis=1) / 
                           enhanced_stats[['train', 'val', 'test']].mean(axis=1)) * 100).round(1)
    
    # Reorder columns for better readability
    column_order = ['Category', 'train', 'val', 'test', 'Total', 'Train %', 'Val %', 'Test %', 
                   'Percentage', 'Class Balance', 'Std_Dev', 'CV']
    enhanced_stats = enhanced_stats[column_order]
    
    # Save enhanced statistics
    enhanced_stats.to_csv('class_counts.csv', index=False)
    
    # 6. Print detailed statistics
    print("\n2. Detailed Class Statistics:")
    print(enhanced_stats.to_string(index=False))
    
    print("\n3. Statistical Summary:")
    print(f"Dataset Balance Analysis:")
    print(f"  - Most represented class: {df.loc[df['Total'].idxmax(), 'Category']} ({df['Total'].max()} images)")
    print(f"  - Least represented class: {df.loc[df['Total'].idxmin(), 'Category']} ({df['Total'].min()} images)")
    print(f"  - Balance ratio: {df['Total'].min()/df['Total'].max():.3f}")
    print(f"  - Standard deviation of class sizes: {df['Total'].std():.1f}")
    print(f"  - Coefficient of variation: {(df['Total'].std()/df['Total'].mean()*100):.1f}%")
    
    # Check for class imbalance
    imbalance_threshold = 0.5  # Classes with less than 50% of max size are considered imbalanced
    imbalanced_classes = df[df['Class Balance'] < imbalance_threshold]
    
    if len(imbalanced_classes) > 0:
        print(f"\n4. Class Imbalance Analysis:")
        print(f"  - {len(imbalanced_classes)} classes have significant imbalance (< 50% of max class size):")
        for _, row in imbalanced_classes.iterrows():
            print(f"    * {row['Category']}: {row['Total']} images ({row['Class Balance']:.3f} ratio)")
    else:
        print(f"\n4. Class Balance: Good - All classes have > 50% of max class size")
    
    print(f"\n✅ Sample statistics generation complete!")
    print(f"📊 Visualizations saved to plots/ directory:")
    print(f"   - class_distribution.png (horizontal bar chart)")
    print(f"   - class_distribution_pie.png (pie chart)")
    print(f"   - split_distribution.png (train/val/test split by category)")
    print(f"📈 Enhanced statistics saved to: class_counts.csv")

if __name__ == "__main__":
    generate_sample_statistics()
