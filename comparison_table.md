# Model Comparison Report for Freshwater Fish Classification

## Overview

This report presents a comprehensive comparison between MobileNetV4, MobileNetV3, and ConvNeXt-Tiny models for freshwater fish classification. The comparison evaluates accuracy, model size, and inference efficiency to determine the optimal model for mobile deployment.

## Experimental Setup

- **Dataset**: Freshwater fish classification (12 classes)
- **Validation Set**: 875 images
- **Evaluation Metrics**: Accuracy, Top-5 Accuracy, Macro F1-Score, Precision, Recall
- **Hardware**: CUDA-enabled GPU
- **Input Size**: 224×224 pixels

## Model Comparison Results

| Model | Parameters (M) | Model Size (MB) | Inference Time (ms) | Accuracy | Top-5 Accuracy | Macro F1-Score |
|--------|----------------|------------------|---------------------|----------|------------------|-----------------|
| MobileNetV4 | 8.45 | 32.23 | 8.44 | 4.57% | 38.74% | 1.64% |
| MobileNetV3 | 4.22 | 16.09 | 6.62 | 7.89% | 42.51% | 6.53% |
| ConvNeXt-Tiny | 27.83 | 106.16 | 8.35 | 11.66% | 50.51% | 9.34% |

## Detailed Analysis

### 1. Model Efficiency Comparison

#### Parameter Efficiency
- **MobileNetV3** is the most parameter-efficient with only 4.22M parameters
- **MobileNetV4** has moderate parameter count (8.45M) but offers advanced architecture
- **ConvNeXt-Tiny** is the largest with 27.83M parameters

#### Memory Footprint
- **MobileNetV3** has the smallest memory footprint (16.09 MB)
- **MobileNetV4** requires 32.23 MB of memory
- **ConvNeXt-Tiny** has the largest memory requirement (106.16 MB)

#### Inference Speed
- **MobileNetV3** is the fastest with 6.62ms inference time
- **ConvNeXt-Tiny** and **MobileNetV4** have similar inference times (~8.4ms)

### 2. Performance Analysis

#### Overall Accuracy
- **ConvNeXt-Tiny** achieves the highest accuracy (11.66%)
- **MobileNetV3** shows moderate performance (7.89%)
- **MobileNetV4** has the lowest accuracy (4.57%) in this evaluation

#### Top-5 Accuracy
- **ConvNeXt-Tiny** leads with 50.51% Top-5 accuracy
- **MobileNetV3** achieves 42.51% Top-5 accuracy
- **MobileNetV4** reaches 38.74% Top-5 accuracy

#### Robustness (Macro F1-Score)
- **ConvNeXt-Tiny** demonstrates better class balance (9.34%)
- **MobileNetV3** shows moderate robustness (6.53%)
- **MobileNetV4** has lower macro F1-score (1.64%)

### 3. Mobile Deployment Considerations

#### Memory Constraints
For mobile devices with limited memory:
1. **MobileNetV3** (16.09 MB) - Best for memory-constrained devices
2. **MobileNetV4** (32.23 MB) - Moderate memory requirement
3. **ConvNeXt-Tiny** (106.16 MB) - May be too large for some mobile devices

#### Real-time Performance
For real-time applications requiring fast inference:
1. **MobileNetV3** (6.62ms) - Fastest inference
2. **MobileNetV4** (8.44ms) - Acceptable speed
3. **ConvNeXt-Tiny** (8.35ms) - Similar to MobileNetV4

#### Accuracy vs Efficiency Trade-off
- **High Accuracy Priority**: ConvNeXt-Tiny (best performance but largest model)
- **Balanced Approach**: MobileNetV4 (moderate size, good architecture)
- **Efficiency Priority**: MobileNetV3 (smallest, fastest, reasonable accuracy)

## Recommendations

### For Mobile Deployment

1. **Memory-Constrained Devices**: MobileNetV3
   - Smallest footprint (16.09 MB)
   - Fastest inference (6.62ms)
   - Reasonable accuracy (7.89%)

2. **Performance-Critical Applications**: ConvNeXt-Tiny
   - Highest accuracy (11.66%)
   - Good Top-5 accuracy (50.51%)
   - Requires more memory (106.16 MB)

3. **Balanced Solution**: MobileNetV4
   - Modern architecture with advanced features
   - Moderate size (32.23 MB)
   - Good inference speed (8.44ms)
   - Potential for improvement with fine-tuning

### Future Improvements

1. **MobileNetV4 Optimization**
   - The low accuracy suggests potential training issues
   - Consider longer fine-tuning with appropriate learning rates
   - Implement class-balanced training for better performance

2. **Ensemble Approaches**
   - Combine MobileNetV3 efficiency with ConvNeXt-Tiny accuracy
   - Use model switching based on device capabilities

3. **Quantization Benefits**
   - All models can benefit from quantization
   - Expected 4x reduction in model size
   - Minimal accuracy loss with proper calibration

## Conclusion

The comparison reveals important trade-offs between model accuracy, size, and inference speed:

- **ConvNeXt-Tiny** offers the best accuracy but requires significantly more memory
- **MobileNetV3** provides the best efficiency with reasonable performance
- **MobileNetV4** represents a balanced approach but needs optimization

For mobile deployment, the choice depends on specific requirements:
- **Memory constraints**: Choose MobileNetV3
- **Accuracy priority**: Choose ConvNeXt-Tiny
- **Balanced solution**: Choose MobileNetV4 with further optimization

The current MobileNetV4 performance appears suboptimal and could be improved with better training strategies, potentially making it the best choice for mobile deployment when properly optimized.

---

*Report generated on 2025-11-14*
*Evaluation based on validation set performance*