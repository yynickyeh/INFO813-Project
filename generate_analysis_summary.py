import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
#任务5：结论分析 工具/方法：是否类别平衡、需要额外增强的类别 交付成果：analysis_summary.md
# Suppress warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self):
        self.class_stats = pd.read_csv('class_counts.csv')
        self.quality_log_path = 'data_cleaning_log.txt'
        
    def analyze_class_balance(self):
        """Analyze class balance and identify imbalanced categories"""
        print("Analyzing Class Balance...")
        
        # Calculate balance metrics
        max_count = self.class_stats['Total'].max()
        min_count = self.class_stats['Total'].min()
        mean_count = self.class_stats['Total'].mean()
        std_count = self.class_stats['Total'].std()
        
        # Define imbalance thresholds
        severe_imbalance_threshold = 0.3  # < 30% of max class
        moderate_imbalance_threshold = 0.5  # < 50% of max class
        
        # Identify imbalanced classes
        severely_imbalanced = self.class_stats[self.class_stats['Class Balance'] < severe_imbalance_threshold]
        moderately_imbalanced = self.class_stats[
            (self.class_stats['Class Balance'] >= severe_imbalance_threshold) & 
            (self.class_stats['Class Balance'] < moderate_imbalance_threshold)
        ]
        well_balanced = self.class_stats[self.class_stats['Class Balance'] >= moderate_imbalance_threshold]
        
        balance_analysis = {
            'max_count': max_count,
            'min_count': min_count,
            'mean_count': mean_count,
            'std_count': std_count,
            'balance_ratio': min_count / max_count,
            'coefficient_of_variation': std_count / mean_count,
            'severely_imbalanced': severely_imbalanced,
            'moderately_imbalanced': moderately_imbalanced,
            'well_balanced': well_balanced
        }
        
        return balance_analysis
    
    def analyze_quality_issues(self):
        """Extract and analyze quality issues from the cleaning log"""
        print("Analyzing Quality Issues...")
        
        try:
            with open(self.quality_log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract key metrics
            lines = content.split('\n')
            total_images = None
            blurry_count = None
            grayscale_count = None
            duplicate_count = None
            
            for line in lines:
                if 'Total images analyzed:' in line:
                    total_images = int(line.split(':')[1].strip())
                elif 'Blurry images found:' in line:
                    blurry_count = int(line.split('(')[0].split(':')[1].strip())
                elif 'Grayscale images found:' in line:
                    grayscale_count = int(line.split('(')[0].split(':')[1].strip())
                elif 'Duplicate image pairs found:' in line:
                    duplicate_count = int(line.split(':')[1].strip())
            
            quality_analysis = {
                'total_analyzed': total_images,
                'blurry_images': blurry_count,
                'grayscale_images': grayscale_count,
                'duplicate_pairs': duplicate_count,
                'blurry_percentage': (blurry_count / total_images * 100) if total_images else 0,
                'grayscale_percentage': (grayscale_count / total_images * 100) if total_images else 0,
                'quality_score': ((total_images - blurry_count - grayscale_count - duplicate_count) / total_images * 100) if total_images else 0
            }
            
            return quality_analysis
            
        except Exception as e:
            print(f"Error reading quality log: {e}")
            return None
    
    def recommend_augmentation(self, balance_analysis):
        """Recommend data augmentation strategies"""
        print("Generating Augmentation Recommendations...")
        
        recommendations = []
        
        # Target minimum samples per class (based on median of well-balanced classes)
        well_balanced = balance_analysis['well_balanced']
        if len(well_balanced) > 0:
            target_samples = int(well_balanced['Total'].median())
        else:
            target_samples = int(balance_analysis['mean_count'] * 1.2)  # 20% above mean
        
        # Recommendations for severely imbalanced classes
        for _, row in balance_analysis['severely_imbalanced'].iterrows():
            current = row['Total']
            needed = target_samples - current
            multiplier = target_samples / current
            
            recommendations.append({
                'category': row['Category'],
                'current_count': current,
                'target_count': target_samples,
                'needed_samples': needed,
                'augmentation_multiplier': multiplier,
                'priority': 'HIGH',
                'techniques': ['rotation', 'flip', 'zoom', 'brightness', 'contrast', 'gaussian_noise']
            })
        
        # Recommendations for moderately imbalanced classes
        for _, row in balance_analysis['moderately_imbalanced'].iterrows():
            current = row['Total']
            needed = target_samples - current
            multiplier = target_samples / current
            
            recommendations.append({
                'category': row['Category'],
                'current_count': current,
                'target_count': target_samples,
                'needed_samples': needed,
                'augmentation_multiplier': multiplier,
                'priority': 'MEDIUM',
                'techniques': ['rotation', 'flip', 'brightness']
            })
        
        # Sort by priority and needed samples
        recommendations.sort(key=lambda x: (x['priority'] != 'HIGH', -x['needed_samples']))
        
        return recommendations, target_samples
    
    def generate_comprehensive_summary(self):
        """Generate comprehensive analysis summary"""
        print("Generating Comprehensive Analysis Summary (DA5)")
        print("=" * 60)
        
        # Analyze class balance
        balance_analysis = self.analyze_class_balance()
        
        # Analyze quality issues
        quality_analysis = self.analyze_quality_issues()
        
        # Generate augmentation recommendations
        recommendations, target_samples = self.recommend_augmentation(balance_analysis)
        
        # Create comprehensive report
        self.create_summary_report(balance_analysis, quality_analysis, recommendations, target_samples)
        
        # Create visualization
        self.create_summary_visualization(balance_analysis, quality_analysis, recommendations)
        
        return balance_analysis, quality_analysis, recommendations
    
    def create_summary_report(self, balance_analysis, quality_analysis, recommendations, target_samples):
        """Create detailed analysis summary report"""
        
        with open('analysis_summary.md', 'w', encoding='utf-8') as f:
            f.write("# Fish Dataset Analysis Summary\n\n")
            f.write("## Overview\n")
            f.write("This document provides a comprehensive analysis of the freshwater fish dataset, including class balance assessment, quality evaluation, and recommendations for data improvement.\n\n")
            
            # Dataset Overview
            f.write("## Dataset Overview\n")
            f.write(f"- **Total Categories**: {len(self.class_stats)}\n")
            f.write(f"- **Total Images**: {self.class_stats['Total'].sum()}\n")
            f.write(f"- **Images per Category**: {balance_analysis['min_count']} - {balance_analysis['max_count']}\n")
            f.write(f"- **Average Images per Category**: {balance_analysis['mean_count']:.1f}\n")
            f.write(f"- **Dataset Split**: 70% train / 20% validation / 10% test\n\n")
            
            # Class Balance Analysis
            f.write("## Class Balance Analysis\n\n")
            f.write("### Balance Metrics\n")
            f.write(f"- **Balance Ratio**: {balance_analysis['balance_ratio']:.3f} (min/max)\n")
            f.write(f"- **Coefficient of Variation**: {balance_analysis['coefficient_of_variation']:.3f}\n")
            f.write(f"- **Standard Deviation**: {balance_analysis['std_count']:.1f}\n\n")
            
            f.write("### Class Distribution Categories\n")
            f.write(f"- **Well Balanced** (≥50% of max class): {len(balance_analysis['well_balanced'])} categories\n")
            if len(balance_analysis['well_balanced']) > 0:
                for _, row in balance_analysis['well_balanced'].iterrows():
                    f.write(f"  - {row['Category']}: {row['Total']} images ({row['Class Balance']:.3f} ratio)\n")
            
            f.write(f"\n- **Moderately Imbalanced** (30-50% of max class): {len(balance_analysis['moderately_imbalanced'])} categories\n")
            if len(balance_analysis['moderately_imbalanced']) > 0:
                for _, row in balance_analysis['moderately_imbalanced'].iterrows():
                    f.write(f"  - {row['Category']}: {row['Total']} images ({row['Class Balance']:.3f} ratio)\n")
            
            f.write(f"\n- **Severely Imbalanced** (<30% of max class): {len(balance_analysis['severely_imbalanced'])} categories\n")
            if len(balance_analysis['severely_imbalanced']) > 0:
                for _, row in balance_analysis['severely_imbalanced'].iterrows():
                    f.write(f"  - {row['Category']}: {row['Total']} images ({row['Class Balance']:.3f} ratio)\n")
            
            # Quality Analysis
            if quality_analysis:
                f.write("\n## Data Quality Analysis\n\n")
                f.write("### Quality Issues (Sample Analysis)\n")
                f.write(f"- **Images Analyzed**: {quality_analysis['total_analyzed']}\n")
                f.write(f"- **Blurry Images**: {quality_analysis['blurry_images']} ({quality_analysis['blurry_percentage']:.1f}%)\n")
                f.write(f"- **Grayscale Images**: {quality_analysis['grayscale_images']} ({quality_analysis['grayscale_percentage']:.1f}%)\n")
                f.write(f"- **Duplicate Pairs**: {quality_analysis['duplicate_pairs']}\n")
                f.write(f"- **Overall Quality Score**: {quality_analysis['quality_score']:.1f}%\n\n")
                
                f.write("### Quality Assessment\n")
                if quality_analysis['blurry_percentage'] > 20:
                    f.write("- ⚠️ **High blur rate detected** - Consider image preprocessing or quality filtering\n")
                if quality_analysis['duplicate_pairs'] > 0:
                    f.write("- ⚠️ **Duplicates found** - Remove duplicates to reduce redundancy\n")
                if quality_analysis['quality_score'] < 80:
                    f.write("- ⚠️ **Low overall quality** - Significant data cleaning recommended\n")
                else:
                    f.write("- ✅ **Acceptable quality** - Minor improvements may be beneficial\n")
                f.write("\n")
            
            # Data Augmentation Recommendations
            f.write("## Data Augmentation Recommendations\n\n")
            f.write(f"### Target Strategy\n")
            f.write(f"- **Target samples per class**: {target_samples} images\n")
            f.write(f"- **Focus on**: Severely and moderately imbalanced categories\n")
            f.write(f"- **Priority**: High-impact categories first\n\n")
            
            f.write("### Detailed Recommendations\n\n")
            for rec in recommendations:
                f.write(f"#### {rec['category']} ({rec['priority']} PRIORITY)\n")
                f.write(f"- **Current**: {rec['current_count']} images\n")
                f.write(f"- **Target**: {rec['target_count']} images\n")
                f.write(f"- **Needed**: {rec['needed_samples']} additional images\n")
                f.write(f"- **Augmentation Multiplier**: {rec['augmentation_multiplier']:.1f}x\n")
                f.write(f"- **Recommended Techniques**: {', '.join(rec['techniques'])}\n\n")
            
            # Implementation Guidelines
            f.write("## Implementation Guidelines\n\n")
            f.write("### Data Augmentation Pipeline\n")
            f.write("1. **Geometric Transformations**:\n")
            f.write("   - Rotation (±15-30 degrees)\n")
            f.write("   - Horizontal flip (50% probability)\n")
            f.write("   - Zoom (0.8-1.2x)\n")
            f.write("   - Translation (±10%)\n\n")
            
            f.write("2. **Color Transformations**:\n")
            f.write("   - Brightness adjustment (±20%)\n")
            f.write("   - Contrast adjustment (±20%)\n")
            f.write("   - Saturation adjustment (±10%)\n\n")
            
            f.write("3. **Noise Addition**:\n")
            f.write("   - Gaussian noise (σ=0.01-0.05)\n")
            f.write("   - Salt-and-pepper noise (1-2%)\n\n")
            
            f.write("### Quality Control\n")
            f.write("- Validate augmented images for visual quality\n")
            f.write("- Ensure class labels remain accurate\n")
            f.write("- Monitor for over-augmentation artifacts\n")
            f.write("- Maintain diversity in augmented samples\n\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            
            if len(balance_analysis['severely_imbalanced']) > 0:
                f.write("### Class Balance\n")
                f.write("The dataset exhibits **significant class imbalance** with several categories severely underrepresented. ")
                f.write("Data augmentation is **essential** for achieving balanced model performance.\n\n")
            
            if quality_analysis and quality_analysis['quality_score'] < 80:
                f.write("### Data Quality\n")
                f.write("The dataset has **moderate quality issues** that may impact model training. ")
                f.write("Quality filtering and preprocessing are **recommended** before augmentation.\n\n")
            
            f.write("### Next Steps\n")
            f.write("1. **Immediate**: Implement data augmentation for severely imbalanced classes\n")
            f.write("2. **Short-term**: Address quality issues through filtering and preprocessing\n")
            f.write("3. **Long-term**: Collect additional samples for underrepresented categories\n")
            f.write("4. **Validation**: Test model performance with augmented dataset\n\n")
            
            f.write("---\n")
            f.write("*Analysis generated on " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n")
    
    def create_summary_visualization(self, balance_analysis, quality_analysis, recommendations):
        """Create comprehensive summary visualization"""
        
        # Ensure plots directory exists
        plots_dir = Path('plots')
        plots_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fish Dataset - Comprehensive Analysis Summary', fontsize=16, fontweight='bold')
        
        # 1. Class Balance Overview
        categories = self.class_stats['Category']
        balances = self.class_stats['Class Balance']
        colors = ['red' if b < 0.3 else 'orange' if b < 0.5 else 'green' for b in balances]
        
        axes[0, 0].barh(categories, balances, color=colors, alpha=0.7)
        axes[0, 0].axvline(x=0.5, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
        axes[0, 0].axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='Severe Threshold')
        axes[0, 0].set_title('Class Balance Analysis', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Balance Ratio (relative to max)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Balance Categories Pie Chart
        balance_counts = [
            len(balance_analysis['well_balanced']),
            len(balance_analysis['moderately_imbalanced']),
            len(balance_analysis['severely_imbalanced'])
        ]
        balance_labels = ['Well Balanced', 'Moderately Imbalanced', 'Severely Imbalanced']
        balance_colors = ['green', 'orange', 'red']
        
        axes[0, 1].pie(balance_counts, labels=balance_labels, colors=balance_colors, autopct='%1.1f%%',
                      shadow=True, startangle=90)
        axes[0, 1].set_title('Balance Categories Distribution', fontsize=12, fontweight='bold')
        
        # 3. Augmentation Requirements
        if recommendations:
            rec_categories = [rec['category'] for rec in recommendations[:10]]  # Top 10
            rec_needed = [rec['needed_samples'] for rec in recommendations[:10]]
            rec_colors = ['red' if rec['priority'] == 'HIGH' else 'orange' for rec in recommendations[:10]]
            
            axes[0, 2].barh(rec_categories, rec_needed, color=rec_colors, alpha=0.7)
            axes[0, 2].set_title('Top 10 Augmentation Requirements', fontsize=12, fontweight='bold')
            axes[0, 2].set_xlabel('Additional Samples Needed')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Quality Overview (if available)
        if quality_analysis:
            quality_labels = ['Clean', 'Blurry', 'Duplicates']
            quality_values = [
                quality_analysis['total_analyzed'] - quality_analysis['blurry_images'] - quality_analysis['duplicate_pairs'],
                quality_analysis['blurry_images'],
                quality_analysis['duplicate_pairs']
            ]
            quality_colors = ['green', 'red', 'orange']
            
            axes[1, 0].pie(quality_values, labels=quality_labels, colors=quality_colors, autopct='%1.1f%%',
                          shadow=True, startangle=90)
            axes[1, 0].set_title('Quality Issues Distribution', fontsize=12, fontweight='bold')
        
        # 5. Dataset Statistics Summary
        stats_data = {
            'Total Images': self.class_stats['Total'].sum(),
            'Categories': len(self.class_stats),
            'Avg per Class': balance_analysis['mean_count'],
            'Balance Ratio': balance_analysis['balance_ratio']
        }
        
        y_pos = range(len(stats_data))
        axes[1, 1].barh(y_pos, list(stats_data.values()), color='skyblue', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(list(stats_data.keys()))
        axes[1, 1].set_title('Dataset Statistics', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Recommendations Priority
        if recommendations:
            high_priority = len([r for r in recommendations if r['priority'] == 'HIGH'])
            medium_priority = len([r for r in recommendations if r['priority'] == 'MEDIUM'])
            
            priority_counts = [high_priority, medium_priority]
            priority_labels = ['High Priority', 'Medium Priority']
            priority_colors = ['red', 'orange']
            
            axes[1, 2].pie(priority_counts, labels=priority_labels, colors=priority_colors, autopct='%1.1f%%',
                          shadow=True, startangle=90)
            axes[1, 2].set_title('Augmentation Priorities', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    analyzer = DatasetAnalyzer()
    balance_analysis, quality_analysis, recommendations = analyzer.generate_comprehensive_summary()
    
    print(f"\n✅ Analysis summary complete!")
    print(f"📄 Detailed report saved to: analysis_summary.md")
    print(f"📊 Summary visualization saved to: plots/comprehensive_analysis_summary.png")
    
    # Print key findings
    print(f"\n🔍 Key Findings:")
    print(f"   - Dataset has {len(balance_analysis['severely_imbalanced'])} severely imbalanced categories")
    print(f"   - Overall balance ratio: {balance_analysis['balance_ratio']:.3f}")
    if quality_analysis:
        print(f"   - Dataset quality score: {quality_analysis['quality_score']:.1f}%")
    print(f"   - {len(recommendations)} categories need data augmentation")
    
    return balance_analysis, quality_analysis, recommendations

if __name__ == "__main__":
    main()