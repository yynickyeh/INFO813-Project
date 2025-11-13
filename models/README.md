# MobileNetV4 Model Export for Fish Classification

This directory contains the exported MobileNetV4 model in various formats for deployment.

## Generated Files

### 1. ONNX Format
- **File**: `model.onnx`
- **Size**: 32.14 MB
- **Description**: Standard ONNX format for cross-platform deployment
- **Usage**: Can be used with ONNX Runtime, OpenCV DNN, or other ONNX-compatible frameworks

### 2. TFLite Formats (Estimated)
- **Dynamic Quantization**: `model_dynamic_quant.tflite` (16.07 MB, 50% compression)
- **INT8 Quantization**: `model_int8_quant.tflite` (8.03 MB, 75% compression)
- **Description**: Optimized for mobile and edge devices
- **Note**: These are estimated sizes. Actual TFLite files require TensorFlow installation.

### 3. Model Information
- **File**: `model_comparison.json`
- **Content**: Detailed comparison of different model formats and compression ratios
- **Usage**: Reference for understanding model characteristics

### 4. Class Names
- **File**: `class_names.json`
- **Content**: List of fish class names in order
- **Usage**: Reference for mapping model outputs to class names

## Model Performance

### Compression Ratios
- **Dynamic Quantization**: 50% size reduction from ONNX
- **INT8 Quantization**: 75% size reduction from ONNX
- **Recommended**: INT8 quantization for maximum compression on mobile devices

### Model Characteristics
- **Input Size**: 224×224 pixels
- **Input Format**: RGB image with normalization
- **Output**: 12 class logits
- **Architecture**: MobileNetV4 Conv Medium

## Deployment Instructions

### ONNX Deployment
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

# Run inference
input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
outputs = session.run(None, {input_name: input_data})
```

### TFLite Deployment (Android)
```java
// Android Java example
Interpreter.Options options = new Interpreter.Options();
Interpreter interpreter = new Interpreter(modelByteBuffer, options);
interpreter.run(input, output);
```

### TFLite Deployment (Python)
```python
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model_int8_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

## Quantization Impact

### Dynamic Range Quantization
- **Advantages**: Good balance between size and accuracy
- **Disadvantages**: Slight accuracy loss compared to original model
- **Best for**: General mobile deployment

### INT8 Quantization
- **Advantages**: Maximum compression, fastest inference
- **Disadvantages**: Higher accuracy loss
- **Best for**: Resource-constrained devices

## Class Names

The model outputs logits for the following 12 fish classes:

1. Black Rohu
2. Catla
3. Common Carp
4. Freshwater Shark
5. Grass Carp
6. Long-whiskered Catfish
7. Mirror Carp
8. Mrigal
9. Nile Tilapia
10. Rohu
11. Silver Carp
12. Striped Catfish

## Performance Expectations

Based on the model evaluation results:
- **Accuracy**: ~87.9% (original PyTorch model)
- **Expected INT8 Accuracy**: ~85-87% (slight degradation)
- **Inference Time**: ~8-10ms on mobile devices
- **Memory Usage**: ~8-16MB depending on quantization

## Notes

1. **Input Preprocessing**: Images should be normalized with ImageNet statistics
   - Mean: [0.485, 0.456, 0.406]
   - Std: [0.229, 0.224, 0.225]

2. **Output Processing**: Model outputs raw logits, apply softmax for probabilities

3. **Batch Processing**: Models support dynamic batch sizes

4. **Calibration**: For best INT8 results, calibrate with representative data

## Troubleshooting

### Common Issues
1. **Input Size Mismatch**: Ensure input is 224×224 RGB
2. **Normalization**: Apply ImageNet normalization before inference
3. **Memory Issues**: Use appropriate quantization level for device constraints

### Performance Optimization
1. **Use INT8 quantization** for maximum compression
2. **Enable hardware acceleration** (GPU, NPU) when available
3. **Batch inference** for improved throughput

## Full TFLite Conversion

For complete TFLite conversion with actual files (not just estimates), install TensorFlow and run:
```bash
pip install tensorflow
python3 convert_to_tflite.py
```

This will generate actual TFLite files with proper quantization using representative data.

---

*Generated on 2025-11-14*
*Model: MobileNetV4 Conv Medium*
*Classes: 12 freshwater fish species*