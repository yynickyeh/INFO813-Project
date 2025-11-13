import os
import shutil
import random
from pathlib import Path
# 任务1：数据集组织 工具/方法：整理为 train/val/test 目录，比例 70/20/10 交付成果：processed/ 文件夹结构
def organize_dataset():
    # Define paths
    source_dir = Path("BD-Freshwater-Fish")
    target_dir = Path("processed")
    
    # Create target directory structure
    target_dir.mkdir(exist_ok=True)
    
    # Create train/val/test directories
    splits = ['train', 'val', 'test']
    for split in splits:
        (target_dir / split).mkdir(exist_ok=True)
    
    # Get all fish categories
    fish_categories = [d for d in source_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(fish_categories)} fish categories:")
    for category in fish_categories:
        print(f"  - {category.name}")
    
    # Process each fish category
    for category in fish_categories:
        print(f"\nProcessing {category.name}...")
        
        # Create category directories in each split
        for split in splits:
            (target_dir / split / category.name).mkdir(exist_ok=True)
        
        # Find all image files in the nested directory structure
        image_files = []
        for root, dirs, files in os.walk(category):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    image_files.append(Path(root) / file)
        
        print(f"  Found {len(image_files)} images")
        
        # Shuffle files for random distribution
        random.shuffle(image_files)
        
        # Calculate split indices (70/20/10)
        total_files = len(image_files)
        train_count = int(total_files * 0.7)
        val_count = int(total_files * 0.2)
        test_count = total_files - train_count - val_count
        
        print(f"  Split: {train_count} train, {val_count} val, {test_count} test")
        
        # Split files
        train_files = image_files[:train_count]
        val_files = image_files[train_count:train_count + val_count]
        test_files = image_files[train_count + val_count:]
        
        # Copy files to target directories
        for file_path in train_files:
            target_path = target_dir / 'train' / category.name / file_path.name
            shutil.copy2(file_path, target_path)
        
        for file_path in val_files:
            target_path = target_dir / 'val' / category.name / file_path.name
            shutil.copy2(file_path, target_path)
        
        for file_path in test_files:
            target_path = target_dir / 'test' / category.name / file_path.name
            shutil.copy2(file_path, target_path)
        
        print(f"  Copied all files to {category.name} directories")
    
    print("\nDataset organization complete!")
    
    # Print summary
    print("\nSummary:")
    for split in splits:
        total_images = 0
        categories = [d for d in (target_dir / split).iterdir() if d.is_dir()]
        for category in categories:
            images = [f for f in category.iterdir() if f.is_file()]
            total_images += len(images)
        print(f"  {split}: {total_images} total images across {len(categories)} categories")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    organize_dataset()