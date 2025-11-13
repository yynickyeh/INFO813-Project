import os
from pathlib import Path
import pandas as pd
# 任务2：样本统计 工具/方法：统计各类样本数量、比例 交付成果：class_counts.csv
def verify_dataset_split():
    processed_dir = Path("processed")
    splits = ['train', 'val', 'test']
    
    # Collect statistics
    stats = []
    total_images = 0
    
    # Get all categories from train directory
    categories = [d.name for d in (processed_dir / 'train').iterdir() if d.is_dir()]
    
    print("Dataset Split Verification")
    print("=" * 50)
    
    for category in sorted(categories):
        category_stats = {'Category': category}
        category_total = 0
        
        for split in splits:
            category_path = processed_dir / split / category
            if category_path.exists():
                image_count = len([f for f in category_path.iterdir() if f.is_file()])
                category_stats[split] = image_count
                category_total += image_count
            else:
                category_stats[split] = 0
        
        category_stats['Total'] = category_total
        category_stats['Train %'] = round((category_stats['train'] / category_total) * 100, 1) if category_total > 0 else 0
        category_stats['Val %'] = round((category_stats['val'] / category_total) * 100, 1) if category_total > 0 else 0
        category_stats['Test %'] = round((category_stats['test'] / category_total) * 100, 1) if category_total > 0 else 0
        
        stats.append(category_stats)
        total_images += category_total
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(stats)
    df.to_csv('class_counts.csv', index=False)
    
    # Print summary table
    print("\nClass Distribution:")
    print(df.to_string(index=False))
    
    # Calculate overall statistics
    overall_train = df['train'].sum()
    overall_val = df['val'].sum()
    overall_test = df['test'].sum()
    overall_total = df['Total'].sum()
    
    print(f"\nOverall Summary:")
    print(f"Total Images: {overall_total}")
    print(f"Train: {overall_train} ({round((overall_train/overall_total)*100, 1)}%)")
    print(f"Validation: {overall_val} ({round((overall_val/overall_total)*100, 1)}%)")
    print(f"Test: {overall_test} ({round((overall_test/overall_total)*100, 1)}%)")
    
    # Check if ratios are close to 70/20/10
    train_ratio = round((overall_train/overall_total)*100, 1)
    val_ratio = round((overall_val/overall_total)*100, 1)
    test_ratio = round((overall_test/overall_total)*100, 1)
    
    print(f"\nTarget vs Actual Ratios:")
    print(f"Train: 70.0% vs {train_ratio}% (diff: {abs(70.0-train_ratio)}%)")
    print(f"Validation: 20.0% vs {val_ratio}% (diff: {abs(20.0-val_ratio)}%)")
    print(f"Test: 10.0% vs {test_ratio}% (diff: {abs(10.0-test_ratio)}%)")
    
    # Verify directory structure
    print(f"\nDirectory Structure Verification:")
    for split in splits:
        split_path = processed_dir / split
        if split_path.exists():
            category_count = len([d for d in split_path.iterdir() if d.is_dir()])
            print(f"  {split}/: {category_count} categories")
        else:
            print(f"  {split}/: MISSING")
    
    print(f"\n✅ Dataset organization complete!")
    print(f"📁 Processed directory created at: {processed_dir.absolute()}")
    print(f"📊 Class statistics saved to: class_counts.csv")

if __name__ == "__main__":
    verify_dataset_split()