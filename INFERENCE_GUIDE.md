# 图片推理使用指南

本指南介绍如何使用训练好的模型对图片进行分类推理，输出对应的类型结果。

## 功能概述

推理系统支持以下功能：
1. **图片加载和预处理**：自动处理各种格式的图片
2. **多任务预测**：同时输出DeepFashion属性、Fabric类型、Fiber类型
3. **批量处理**：支持批量处理多张图片
4. **结果格式化**：提供易读的结果输出格式
5. **置信度评估**：为每个预测提供置信度分数

## 文件说明

### 核心文件
- **`inference.py`**: 主要的推理类和功能实现
- **`demo_inference.py`**: 命令行演示脚本
- **`test_inference.py`**: 推理功能测试脚本

### 推理类 `FashionInference`

主要方法：
- `__init__(model_path, device)`: 初始化推理器
- `predict(image_input)`: 单张图片推理
- `predict_batch(image_paths)`: 批量图片推理
- `format_results(results, detailed)`: 格式化结果输出

## 使用方法

### 1. 基本使用

```python
from inference import FashionInference

# 创建推理器
inferencer = FashionInference("mixed_checkpoints/best_model.pth")

# 单张图片推理
results = inferencer.predict("path/to/image.jpg")

# 格式化输出
formatted = inferencer.format_results(results, detailed=True)
print(formatted)
```

### 2. 命令行使用

#### 单张图片推理
```bash
python demo_inference.py --image path/to/image.jpg --detailed
```

#### 批量图片推理
```bash
python demo_inference.py --batch image1.jpg image2.jpg image3.jpg
```

#### 自动寻找测试图片
```bash
python demo_inference.py --auto-find --detailed
```

#### 保存结果到文件
```bash
python demo_inference.py --auto-find --output results.json
```

### 3. 批量处理

```python
# 批量处理多张图片
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_results = inferencer.predict_batch(image_paths)

for result in batch_results:
    if 'error' not in result:
        print(f"图片: {result['image_path']}")
        formatted = inferencer.format_results(result)
        print(formatted)
```

## 输出结果说明

### 结果结构

推理结果包含以下主要部分：

```python
{
    'raw_outputs': {          # 原始模型输出
        'deepfashion_attrs': tensor,
        'fabric': tensor,
        'fiber': tensor,
        'textile': tensor
    },
    'predictions': {          # 解析后的预测结果
        'deepfashion_attrs': [...],
        'fabric': {...},
        'fiber': {...},
        'textile': {...}
    },
    'probabilities': {        # 概率分布
        'deepfashion_attrs': tensor,
        'fabric': tensor,
        'fiber': tensor
    },
    'top_predictions': {      # Top-K预测
        'fabric': [...],
        'fiber': [...]
    }
}
```

### 1. DeepFashion属性预测

输出激活的属性及其置信度：

```python
'deepfashion_attrs': [
    {'attribute': 'texture_1', 'confidence': 0.85},
    {'attribute': 'fabric_2', 'confidence': 0.72},
    {'attribute': 'style_3', 'confidence': 0.68}
]
```

### 2. Fabric面料预测

输出最可能的面料类型：

```python
'fabric': {
    'class': 'denim',
    'confidence': 0.92
}
```

Top-5预测：
```python
'fabric': [
    {'class': 'denim', 'confidence': 0.92},
    {'class': 'canvas', 'confidence': 0.05},
    {'class': 'twill', 'confidence': 0.02}
]
```

### 3. Fiber纤维预测

输出最可能的纤维类型：

```python
'fiber': {
    'class': 'cotton',
    'confidence': 0.88
}
```

### 4. 分割结果（如果启用）

```python
'segmentation': {
    'mask': numpy_array,      # 分割掩码
    'coverage': 0.75          # 覆盖率
}
```

## 格式化输出示例

```
============================================================
图片分类结果
============================================================

📋 DeepFashion属性 (5个):
  • texture_1: 0.850
  • fabric_2: 0.720
  • style_3: 0.680
  • part_1: 0.650
  • shape_2: 0.620

🧵 面料类型:
  • denim: 0.920
  Top-5预测:
    - denim: 0.920
    - canvas: 0.050
    - twill: 0.020
    - corduroy: 0.008
    - flannel: 0.002

🧶 纤维类型:
  • cotton: 0.880
  Top-5预测:
    - cotton: 0.880
    - polyester: 0.080
    - wool: 0.025
    - silk: 0.010
    - nylon: 0.005

✂️ 分割结果:
  • 覆盖率: 0.750

============================================================
```

## 支持的图片格式

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)

## 预处理步骤

1. **尺寸调整**: 调整到224x224像素
2. **格式转换**: 转换为RGB格式
3. **张量化**: 转换为PyTorch张量
4. **标准化**: 使用ImageNet标准化参数
5. **设备转移**: 移动到指定设备（CPU/GPU）

## 性能优化建议

### 1. GPU加速
```python
# 使用GPU推理（如果可用）
inferencer = FashionInference(model_path, device="cuda")
```

### 2. 批量处理
```python
# 批量处理比单张处理更高效
batch_results = inferencer.predict_batch(image_paths)
```

### 3. 模型优化
```python
# 可以考虑模型量化或其他优化技术
# 这需要在模型训练或保存时进行
```

## 错误处理

### 常见错误及解决方案

1. **模型文件不存在**
   ```
   错误: 模型文件不存在: path/to/model.pth
   解决: 检查模型路径，确保已完成训练
   ```

2. **图片文件损坏**
   ```
   错误: 图片加载失败
   解决: 检查图片文件是否完整，格式是否支持
   ```

3. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 使用CPU推理或减少批量大小
   ```

4. **类别信息缺失**
   ```
   警告: 无法加载类别信息
   解决: 确保数据集目录结构正确
   ```

## 测试和验证

### 运行测试
```bash
# 运行完整测试
python test_inference.py

# 运行演示
python demo_inference.py --auto-find
```

### 验证结果
1. 检查输出格式是否正确
2. 验证置信度分数是否合理
3. 确认类别预测是否符合预期
4. 测试批量处理功能

## 集成到其他项目

### 作为模块使用
```python
import sys
sys.path.append('/path/to/cv_model')

from inference import FashionInference

# 在你的项目中使用
inferencer = FashionInference("model.pth")
results = inferencer.predict("image.jpg")
```

### API封装示例
```python
from flask import Flask, request, jsonify
from inference import FashionInference

app = Flask(__name__)
inferencer = FashionInference("model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    # 保存临时文件并推理
    # ... 处理逻辑 ...
    
    results = inferencer.predict(temp_path)
    return jsonify(results['predictions'])
```

## 注意事项

1. **模型兼容性**: 确保使用的模型与当前代码版本兼容
2. **设备内存**: 大模型可能需要大量GPU内存
3. **图片质量**: 图片质量会影响预测准确性
4. **类别映射**: 确保类别名称映射正确
5. **版本依赖**: 注意PyTorch和其他依赖的版本兼容性

通过这个推理系统，您可以轻松地将训练好的模型应用到实际的图片分类任务中，获得准确的类型预测结果。
