import os
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import hashlib
import imagehash
from skimage.metrics import structural_similarity as ssim
import warnings
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
#任务4：数据质量检查 工具/方法：检测模糊图、灰度图、重复图像（用 SSIM 或哈希比对） 交付成果：data_cleaning_log.txt
# Suppress warnings
warnings.filterwarnings('ignore')

class DataQualityChecker:
    def __init__(self, dataset_path='processed'):
        self.dataset_path = Path(dataset_path)
        self.quality_issues = {
            'blurry_images': [],
            'grayscale_images': [],
            'duplicate_images': [],
            'corrupted_images': []
        }
        self.image_stats = []
        
    def calculate_blur_score(self, image_path):
        """Calculate blur score using Laplacian variance method"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            return blur_score
        except Exception as e:
            print(f"Error calculating blur for {image_path}: {e}")
            return None
    
    def is_grayscale(self, image_path):
        """Check if image is grayscale"""
        try:
            img = Image.open(image_path)
            if img.mode == 'L':
                return True
            
            # Convert to RGB if needed
            img = img.convert('RGB')
            img_array = np.array(img)
            
            # Check if all channels are the same
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            return np.array_equal(r, g) and np.array_equal(g, b)
        except Exception as e:
            print(f"Error checking grayscale for {image_path}: {e}")
            return False
    
    def calculate_phash(self, image_path):
        """Calculate perceptual hash for duplicate detection"""
        try:
            img = Image.open(image_path)
            img = img.convert('RGB')
            phash = imagehash.phash(img)
            return phash
        except Exception as e:
            print(f"Error calculating phash for {image_path}: {e}")
            return None
    
    def calculate_ssim(self, img1_path, img2_path):
        """Calculate Structural Similarity Index between two images"""
        try:
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            if img1 is None or img2 is None:
                return None
            
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Resize to same dimensions
            h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
            gray1 = cv2.resize(gray1, (w, h))
            gray2 = cv2.resize(gray2, (w, h))
            
            # Calculate SSIM
            similarity = ssim(gray1, gray2)
            return similarity
        except Exception as e:
            print(f"Error calculating SSIM between {img1_path} and {img2_path}: {e}")
            return None
    
    def check_image_quality(self, image_path, blur_threshold=100.0):
        """Check individual image quality"""
        try:
            # Basic image info
            img = Image.open(image_path)
            width, height = img.size
            file_size = image_path.stat().st_size / 1024  # KB
            
            # Blur detection
            blur_score = self.calculate_blur_score(image_path)
            is_blurry = blur_score is not None and blur_score < blur_threshold
            
            # Grayscale detection
            is_grayscale = self.is_grayscale(image_path)
            
            # Calculate hash for duplicate detection
            phash = self.calculate_phash(image_path)
            
            return {
                'path': str(image_path),
                'width': width,
                'height': height,
                'file_size_kb': file_size,
                'blur_score': blur_score,
                'is_blurry': is_blurry,
                'is_grayscale': is_grayscale,
                'phash': str(phash) if phash else None
            }
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def find_duplicates(self, image_stats, ssim_threshold=0.95, hash_threshold=5):
        """Find duplicate images using both SSIM and perceptual hash"""
        duplicates = []
        processed_images = []
        
        print("  Checking for duplicates...")
        
        for i, img1 in enumerate(tqdm(image_stats, desc="    Comparing images")):
            if img1['phash'] is None:
                continue
                
            for j, img2 in enumerate(processed_images):
                if img2['phash'] is None:
                    continue
                
                # Check hash distance first (faster)
                try:
                    hash1 = imagehash.hex_to_hash(img1['phash'])
                    hash2 = imagehash.hex_to_hash(img2['phash'])
                    hash_distance = hash1 - hash2
                    
                    if hash_distance <= hash_threshold:
                        # If hashes are similar, check SSIM
                        ssim_score = self.calculate_ssim(img1['path'], img2['path'])
                        
                        if ssim_score is not None and ssim_score >= ssim_threshold:
                            duplicates.append({
                                'image1': img1['path'],
                                'image2': img2['path'],
                                'ssim_score': ssim_score,
                                'hash_distance': hash_distance
                            })
                except Exception as e:
                    continue
            
            processed_images.append(img1)
        
        return duplicates
    
    def run_quality_check(self, sample_size=None, blur_threshold=100.0):
        """Run complete data quality check"""
        print("Starting Data Quality Check (DA4)")
        print("=" * 50)
        
        # Get all image files
        all_images = []
        for split in ['train', 'val', 'test']:
            split_path = self.dataset_path / split
            if split_path.exists():
                for category in split_path.iterdir():
                    if category.is_dir():
                        for img_file in category.iterdir():
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                all_images.append(img_file)
        
        print(f"Found {len(all_images)} total images")
        
        # Sample images if specified
        if sample_size and sample_size < len(all_images):
            all_images = random.sample(all_images, sample_size)
            print(f"Sampling {len(all_images)} images for analysis")
        
        # Check individual image quality
        print("\n1. Checking individual image quality...")
        image_stats = []
        
        for img_path in tqdm(all_images, desc="    Processing images"):
            stats = self.check_image_quality(img_path, blur_threshold)
            if stats:
                image_stats.append(stats)
                
                # Collect quality issues
                if stats['is_blurry']:
                    self.quality_issues['blurry_images'].append({
                        'path': stats['path'],
                        'blur_score': stats['blur_score']
                    })
                
                if stats['is_grayscale']:
                    self.quality_issues['grayscale_images'].append(stats['path'])
        
        self.image_stats = image_stats
        
        # Find duplicates
        print("\n2. Finding duplicate images...")
        duplicates = self.find_duplicates(image_stats)
        self.quality_issues['duplicate_images'] = duplicates
        
        # Generate report
        self.generate_quality_report()
        
        return self.quality_issues
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        print("\n3. Generating quality report...")
        
        # Create data cleaning log
        with open('data_cleaning_log.txt', 'w', encoding='utf-8') as f:
            f.write("Data Quality Check Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total images analyzed: {len(self.image_stats)}\n\n")
            
            # Blur detection results
            f.write("1. BLUR DETECTION RESULTS\n")
            f.write("-" * 30 + "\n")
            blurry_count = len(self.quality_issues['blurry_images'])
            f.write(f"Blurry images found: {blurry_count} ({blurry_count/len(self.image_stats)*100:.2f}%)\n\n")
            
            if blurry_count > 0:
                f.write("Blurry images (blur score < 100.0):\n")
                for img in self.quality_issues['blurry_images'][:20]:  # Show first 20
                    f.write(f"  - {img['path']} (blur score: {img['blur_score']:.2f})\n")
                if blurry_count > 20:
                    f.write(f"  ... and {blurry_count - 20} more\n")
                f.write("\n")
            
            # Grayscale detection results
            f.write("2. GRAYSCALE DETECTION RESULTS\n")
            f.write("-" * 35 + "\n")
            grayscale_count = len(self.quality_issues['grayscale_images'])
            f.write(f"Grayscale images found: {grayscale_count} ({grayscale_count/len(self.image_stats)*100:.2f}%)\n\n")
            
            if grayscale_count > 0:
                f.write("Grayscale images:\n")
                for img_path in self.quality_issues['grayscale_images'][:20]:  # Show first 20
                    f.write(f"  - {img_path}\n")
                if grayscale_count > 20:
                    f.write(f"  ... and {grayscale_count - 20} more\n")
                f.write("\n")
            
            # Duplicate detection results
            f.write("3. DUPLICATE DETECTION RESULTS\n")
            f.write("-" * 35 + "\n")
            duplicate_count = len(self.quality_issues['duplicate_images'])
            f.write(f"Duplicate image pairs found: {duplicate_count}\n\n")
            
            if duplicate_count > 0:
                f.write("Duplicate image pairs (SSIM >= 0.95, Hash distance <= 5):\n")
                for dup in self.quality_issues['duplicate_images'][:20]:  # Show first 20
                    f.write(f"  - {dup['image1']}\n")
                    f.write(f"    {dup['image2']} (SSIM: {dup['ssim_score']:.3f}, Hash dist: {dup['hash_distance']})\n")
                if duplicate_count > 20:
                    f.write(f"  ... and {duplicate_count - 20} more pairs\n")
                f.write("\n")
            
            # Summary statistics
            f.write("4. SUMMARY STATISTICS\n")
            f.write("-" * 25 + "\n")
            
            if self.image_stats:
                blur_scores = [img['blur_score'] for img in self.image_stats if img['blur_score'] is not None]
                file_sizes = [img['file_size_kb'] for img in self.image_stats]
                
                f.write(f"Blur score statistics:\n")
                f.write(f"  Mean: {np.mean(blur_scores):.2f}\n")
                f.write(f"  Std: {np.std(blur_scores):.2f}\n")
                f.write(f"  Min: {np.min(blur_scores):.2f}\n")
                f.write(f"  Max: {np.max(blur_scores):.2f}\n\n")
                
                f.write(f"File size statistics:\n")
                f.write(f"  Mean: {np.mean(file_sizes):.1f} KB\n")
                f.write(f"  Std: {np.std(file_sizes):.1f} KB\n")
                f.write(f"  Min: {np.min(file_sizes):.1f} KB\n")
                f.write(f"  Max: {np.max(file_sizes):.1f} KB\n\n")
            
            # Recommendations
            f.write("5. RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            
            if blurry_count > 0:
                f.write(f"• Consider removing or re-capturing {blurry_count} blurry images\n")
            
            if grayscale_count > 0:
                f.write(f"• Consider converting {grayscale_count} grayscale images to RGB or removing them\n")
            
            if duplicate_count > 0:
                f.write(f"• Remove {duplicate_count} duplicate image pairs to reduce redundancy\n")
            
            total_issues = blurry_count + grayscale_count + duplicate_count
            if total_issues == 0:
                f.write("• No significant quality issues found - dataset is clean!\n")
            else:
                f.write(f"• Total quality issues found: {total_issues}\n")
                f.write(f"• Dataset quality score: {((len(self.image_stats) - total_issues) / len(self.image_stats) * 100):.1f}%\n")
        
        # Create quality visualization
        self.create_quality_visualization()
        
        print(f"✅ Data quality check complete!")
        print(f"📄 Detailed report saved to: data_cleaning_log.txt")
        print(f"📊 Quality visualization saved to: plots/data_quality_analysis.png")
        
        # Print summary
        print(f"\n📋 Quality Check Summary:")
        print(f"   - Total images analyzed: {len(self.image_stats)}")
        print(f"   - Blurry images: {blurry_count} ({blurry_count/len(self.image_stats)*100:.2f}%)")
        print(f"   - Grayscale images: {grayscale_count} ({grayscale_count/len(self.image_stats)*100:.2f}%)")
        print(f"   - Duplicate pairs: {duplicate_count}")
        print(f"   - Dataset quality score: {((len(self.image_stats) - total_issues) / len(self.image_stats) * 100):.1f}%")
    
    def create_quality_visualization(self):
        """Create visualization of quality analysis results"""
        # Ensure plots directory exists
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Analysis Results', fontsize=16, fontweight='bold')
        
        # Blur score distribution
        if self.image_stats:
            blur_scores = [img['blur_score'] for img in self.image_stats if img['blur_score'] is not None]
            axes[0, 0].hist(blur_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(x=100, color='red', linestyle='--', label='Blur Threshold (100)')
            axes[0, 0].set_title('Blur Score Distribution', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Blur Score (Laplacian Variance)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Quality issues summary
        issue_counts = [
            len(self.quality_issues['blurry_images']),
            len(self.quality_issues['grayscale_images']),
            len(self.quality_issues['duplicate_images'])
        ]
        issue_labels = ['Blurry Images', 'Grayscale Images', 'Duplicate Pairs']
        colors = ['red', 'orange', 'purple']
        
        axes[0, 1].bar(issue_labels, issue_counts, color=colors, alpha=0.7)
        axes[0, 1].set_title('Quality Issues Summary', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add percentage labels on bars
        for i, count in enumerate(issue_counts):
            percentage = count / len(self.image_stats) * 100
            axes[0, 1].text(i, count + max(issue_counts) * 0.01, f'{count}\n({percentage:.1f}%)', 
                           ha='center', va='bottom', fontsize=10)
        
        # File size vs blur score scatter
        if self.image_stats:
            file_sizes = [img['file_size_kb'] for img in self.image_stats if img['blur_score'] is not None]
            blur_scores_filtered = [img['blur_score'] for img in self.image_stats if img['blur_score'] is not None]
            
            axes[1, 0].scatter(file_sizes, blur_scores_filtered, alpha=0.6, s=20)
            axes[1, 0].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Blur Threshold')
            axes[1, 0].set_title('File Size vs Blur Score', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('File Size (KB)')
            axes[1, 0].set_ylabel('Blur Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Quality score pie chart
        total_issues = sum(issue_counts)
        clean_images = len(self.image_stats) - total_issues
        
        sizes = [clean_images, total_issues]
        labels = ['Clean Images', 'Quality Issues']
        colors = ['green', 'red']
        explode = (0, 0.1)
        
        axes[1, 1].pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                      shadow=True, startangle=90)
        axes[1, 1].set_title('Dataset Quality Overview', fontsize=12, fontweight='bold')
        axes[1, 1].axis('equal')
        
        plt.tight_layout()
        plt.savefig('plots/data_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Initialize quality checker
    checker = DataQualityChecker()
    
    # Run quality check (sample 500 images for faster processing)
    # Set sample_size=None to check all images
    quality_issues = checker.run_quality_check(sample_size=500, blur_threshold=100.0)
    
    return quality_issues

if __name__ == "__main__":
    main()