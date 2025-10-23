# 服装属性推理系统使用指南

本文档介绍如何使用推理包装器对图片进行服装属性分类。

## 目录

1. [功能概述](#功能概述)
2. [快速开始](#快速开始)
3. [详细使用方法](#详细使用方法)
4. [API文档](#api文档)
5. [输出格式](#输出格式)
6. [常见问题](#常见问题)

---

## 功能概述

推理系统提供以下功能：

- ✅ **单张图片推理**：快速对单张图片进行属性分类
- ✅ **批量推理**：高效处理多张图片
- ✅ **属性分类**：识别服装的多个属性（颜色、图案、袖长、面料等）
- ✅ **纹理分类**：识别Fabric和Fiber类型（可选）
- ✅ **分割支持**：输出服装分割掩码（如果模型支持）
- ✅ **可配置阈值**：自定义属性判定阈值
- ✅ **友好输出**：JSON格式输出，包含置信度信息

---

## 快速开始

### 方法1：使用命令行工具

```bash
# 单张图片推理
python quick_inference.py \
    --model checkpoints/best_model.pth \
    --image test_image.jpg \
    --output result.json

# 批量推理
python quick_inference.py \
    --model checkpoints/best_model.pth \
    --images test_images/ \
    --output batch_results.json

# 自定义阈值
python quick_inference.py \
    --model checkpoints/best_model.pth \
    --image test_image.jpg \
    --threshold 0.6
```

### 方法2：在Python代码中使用

```python
import torch
from inference import FashionInferenceWrapper

# 1. 加载模型
model = torch.load('checkpoints/best_model.pth')

# 2. 创建推理包装器
wrapper = FashionInferenceWrapper(
    model=model,
    threshold=0.5
)

# 3. 推理单张图片
result = wrapper.predict_single('test_image.jpg')

# 4. 查看结果
print(wrapper.get_summary(result))
print(f"检测到的属性: {result['attributes']['predicted']}")
```

---

## 详细使用方法

### 1. 单张图片推理

#### 命令行方式

```bash
python quick_inference.py \
    --model checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --threshold 0.5 \
    --output result.json
```

#### Python代码方式

```python
from inference import FashionInferenceWrapper
import torch

# 加载模型
model = torch.load('checkpoints/best_model.pth')

# 创建包装器
wrapper = FashionInferenceWrapper(model=model, threshold=0.5)

# 推理并保存结果
result = wrapper.predict_and_save(
    image_path='test_image.jpg',
    output_path='result.json',
    return_raw=False  # True=包含所有概率值
)

# 打印可读摘要
print(wrapper.get_summary(result))
```

### 2. 批量推理

#### 命令行方式

```bash
python quick_inference.py \
    --model checkpoints/best_model.pth \
    --images path/to/images/ \
    --output batch_results.json
```

#### Python代码方式

```python
from inference import FashionInferenceWrapper
import torch
import glob

# 加载模型
model = torch.load('checkpoints/best_model.pth')
wrapper = FashionInferenceWrapper(model=model)

# 获取所有图片
image_paths = glob.glob('test_images/*.jpg')

# 批量推理
results = wrapper.predict_batch_and_save(
    image_paths=image_paths,
    output_path='batch_results.json',
    return_raw=False
)

# 处理结果
for result in results:
    print(f"\n图片: {result['image_name']}")
    print(f"属性: {result['attributes']['predicted']}")
```

### 3. 使用自定义属性名称

```python
from inference import FashionInferenceWrapper, load_attr_names_from_file
import torch

# 从DeepFashion标注文件加载属性名称
attr_file = "/path/to/list_attr_cloth.txt"
attr_names = load_attr_names_from_file(attr_file)

# 或手动指定
attr_names = [
    'black', 'white', 'floral', 'striped', 'long_sleeve',
    'short_sleeve', 'cotton', 'denim', 'leather', 'loose', ...
]

# 创建包装器
model = torch.load('checkpoints/best_model.pth')
wrapper = FashionInferenceWrapper(
    model=model,
    attr_names=attr_names
)

# 推理
result = wrapper.predict_single('test_image.jpg')
```

### 4. 启用纹理分类

```python
from inference import FashionInferenceWrapper
import torch

# Fabric类别名称
fabric_names = [
    'canvas', 'denim', 'flannel', 'lace', 'leather',
    'linen', 'satin', 'silk', 'velvet', 'wool', ...
]

# Fiber类别名称
fiber_names = [
    'cotton', 'polyester', 'nylon', 'silk', 'wool',
    'linen', 'acrylic', 'rayon', 'spandex', ...
]

# 加载启用了纹理分类的模型
model = torch.load('checkpoints/best_model_textile.pth')

# 创建包装器
wrapper = FashionInferenceWrapper(
    model=model,
    fabric_names=fabric_names,
    fiber_names=fiber_names,
    enable_textile_classification=True
)

# 推理
result = wrapper.predict_single('test_image.jpg', return_raw=True)

# 查看纹理信息
print(f"Fabric: {result['fabric']['class_name']}")
print(f"Fiber: {result['fiber']['class_name']}")
```

### 5. 调整判定阈值

```python
from inference import FashionInferenceWrapper
import torch

model = torch.load('checkpoints/best_model.pth')

# 尝试不同阈值
for threshold in [0.3, 0.5, 0.7]:
    wrapper = FashionInferenceWrapper(
        model=model,
        threshold=threshold
    )
    
    result = wrapper.predict_single('test_image.jpg')
    
    print(f"\n阈值={threshold}")
    print(f"检测到 {result['attributes']['count']} 个属性")
    print(f"属性: {result['attributes']['predicted']}")
```

---

## API文档

### FashionInferenceWrapper 类

主要推理包装器类。

#### 初始化参数

```python
FashionInferenceWrapper(
    model: nn.Module,                          # 训练好的模型
    attr_names: Optional[List[str]] = None,    # 属性名称列表
    fabric_names: Optional[List[str]] = None,  # Fabric类别名称
    fiber_names: Optional[List[str]] = None,   # Fiber类别名称
    threshold: float = 0.5,                    # 判定阈值
    device: Optional[torch.device] = None,     # 运行设备
    img_size: int = 224,                       # 输入图片尺寸
    enable_textile_classification: bool = False # 启用纹理分类
)
```

#### 主要方法

##### predict_single()

推理单张图片。

```python
result = wrapper.predict_single(
    image_path: str,        # 图片路径
    return_raw: bool = False # 是否返回原始数据
) -> Dict
```

##### predict_batch()

批量推理多张图片。

```python
results = wrapper.predict_batch(
    image_paths: List[str],  # 图片路径列表
    return_raw: bool = False # 是否返回原始数据
) -> List[Dict]
```

##### predict_and_save()

推理并保存结果到JSON文件。

```python
wrapper.predict_and_save(
    image_path: str,     # 图片路径
    output_path: str,    # 输出JSON路径
    return_raw: bool = False
)
```

##### get_summary()

生成可读的结果摘要。

```python
summary = wrapper.get_summary(result: Dict) -> str
```

### 辅助函数

#### load_attr_names_from_file()

从DeepFashion标注文件加载属性名称。

```python
from inference import load_attr_names_from_file

attr_names = load_attr_names_from_file(
    'path/to/list_attr_cloth.txt'
)
```

---

## 输出格式

### 基本输出格式（return_raw=False）

```json
{
  "image_path": "test_image.jpg",
  "image_name": "test_image.jpg",
  "attributes": {
    "predicted": ["floral", "long_sleeve", "cotton", "loose"],
    "count": 4,
    "confidence_scores": {
      "floral": 0.89,
      "long_sleeve": 0.92,
      "cotton": 0.76,
      "loose": 0.68
    }
  }
}
```

### 完整输出格式（return_raw=True）

```json
{
  "image_path": "test_image.jpg",
  "image_name": "test_image.jpg",
  "attributes": {
    "predicted": ["floral", "long_sleeve", "cotton"],
    "count": 3,
    "confidence_scores": {
      "floral": 0.89,
      "long_sleeve": 0.92,
      "cotton": 0.76
    },
    "all_scores": {
      "black": 0.12,
      "white": 0.23,
      "floral": 0.89,
      "graphic": 0.15,
      ...
    }
  },
  "fabric": {
    "class_id": 3,
    "class_name": "cotton",
    "confidence": 0.92,
    "all_probs": {
      "canvas": 0.02,
      "denim": 0.05,
      "cotton": 0.92,
      ...
    }
  },
  "fiber": {
    "class_id": 0,
    "class_name": "cotton",
    "confidence": 0.88
  },
  "segmentation": {
    "has_mask": true,
    "mask_shape": [1, 224, 224],
    "coverage_ratio": 0.68
  }
}
```

### 批量推理输出格式

```json
[
  {
    "image_path": "image1.jpg",
    "attributes": {...}
  },
  {
    "image_path": "image2.jpg",
    "attributes": {...}
  },
  ...
]
```

---

## 命令行参数

### quick_inference.py 参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `--model` | str | ✅ | 模型文件路径 |
| `--image` | str | * | 单张图片路径 |
| `--images` | str | * | 图片目录路径（批量推理） |
| `--threshold` | float | ❌ | 判定阈值（默认0.5） |
| `--output` | str | ❌ | 输出JSON路径（默认inference_result.json） |
| `--attr-file` | str | ❌ | 属性定义文件路径 |
| `--device` | str | ❌ | 运行设备（auto/cpu/cuda，默认auto） |
| `--raw` | flag | ❌ | 输出原始数据 |
| `--no-summary` | flag | ❌ | 不打印摘要 |
| `--textile` | flag | ❌ | 启用纹理分类 |

*注：`--image` 和 `--images` 二选一必需

### 使用示例

```bash
# 基础用法
python quick_inference.py --model model.pth --image test.jpg

# 指定阈值和输出
python quick_inference.py \
    --model model.pth \
    --image test.jpg \
    --threshold 0.6 \
    --output my_result.json

# 批量推理
python quick_inference.py \
    --model model.pth \
    --images test_images/ \
    --output batch_results.json

# 使用CPU运行
python quick_inference.py \
    --model model.pth \
    --image test.jpg \
    --device cpu

# 包含原始数据
python quick_inference.py \
    --model model.pth \
    --image test.jpg \
    --raw

# 启用纹理分类
python quick_inference.py \
    --model model_textile.pth \
    --image test.jpg \
    --textile
```

---

## 常见问题

### Q1: 如何选择合适的阈值？

**A:** 阈值决定了属性的判定标准：
- **0.3-0.4**：更宽松，会检测到更多属性（高召回率）
- **0.5**：平衡的选择（推荐）
- **0.6-0.7**：更严格，只保留高置信度的属性（高精确率）

建议先使用默认的0.5，然后根据实际效果调整。

### Q2: 模型推理速度慢怎么办？

**A:** 优化建议：
1. 使用GPU：`--device cuda`
2. 批量推理：使用 `predict_batch()` 而不是多次 `predict_single()`
3. 减少输入尺寸（修改代码中的 `img_size` 参数）
4. 使用模型量化（需要额外配置）

### Q3: 如何添加自定义属性？

**A:** 
```python
# 方法1：直接指定
custom_attrs = ['my_attr1', 'my_attr2', ...]
wrapper = FashionInferenceWrapper(model=model, attr_names=custom_attrs)

# 方法2：从文件加载并修改
from inference import load_attr_names_from_file

attrs = load_attr_names_from_file('list_attr_cloth.txt')
attrs.append('new_custom_attr')
wrapper = FashionInferenceWrapper(model=model, attr_names=attrs)
```

### Q4: 推理结果为空怎么办？

**A:** 可能的原因：
1. 阈值设置过高：尝试降低阈值（如0.3）
2. 图片预处理问题：确保图片格式正确（RGB）
3. 模型未正确加载：检查模型文件路径
4. 图片质量问题：确保图片清晰度足够

### Q5: 如何在Web服务中使用？

**A:** 示例Flask应用：

```python
from flask import Flask, request, jsonify
from inference import FashionInferenceWrapper
import torch

app = Flask(__name__)

# 加载模型（在启动时加载一次）
model = torch.load('checkpoints/best_model.pth')
wrapper = FashionInferenceWrapper(model=model)

@app.route('/predict', methods=['POST'])
def predict():
    # 接收上传的图片
    file = request.files['image']
    file_path = f'/tmp/{file.filename}'
    file.save(file_path)
    
    # 推理
    result = wrapper.predict_single(file_path)
    
    # 返回JSON结果
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Q6: 支持哪些图片格式？

**A:** 支持所有PIL/Pillow支持的格式：
- ✅ JPEG (.jpg, .jpeg)
- ✅ PNG (.png)
- ✅ BMP (.bmp)
- ✅ TIFF (.tiff, .tif)
- ✅ WebP (.webp)

### Q7: 如何处理大批量图片？

**A:** 对于数千张图片：

```python
from inference import FashionInferenceWrapper
import torch
import glob

model = torch.load('checkpoints/best_model.pth')
wrapper = FashionInferenceWrapper(model=model)

# 获取所有图片
all_images = glob.glob('large_dataset/**/*.jpg', recursive=True)

# 分批处理（每次32张）
batch_size = 32
results = []

for i in range(0, len(all_images), batch_size):
    batch = all_images[i:i+batch_size]
    batch_results = wrapper.predict_batch(batch)
    results.extend(batch_results)
    
    print(f"已处理: {min(i+batch_size, len(all_images))}/{len(all_images)}")

# 保存结果
import json
with open('large_batch_results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 技术支持

如有问题或建议，请查看：
- `inference_example.py` - 更多使用示例
- `base_model.py` - 模型架构
- `training.py` - 训练代码

---

## 更新日志

### v1.0.0 (2024-10-23)
- ✨ 初始版本发布
- ✅ 支持单张和批量推理
- ✅ 支持属性分类
- ✅ 支持纹理分类（可选）
- ✅ 命令行工具
- ✅ 完整的API文档
