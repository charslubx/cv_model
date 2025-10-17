# 多标签分类指南

## 什么是多标签分类？

多标签分类（Multi-Label Classification）是指一个样本可以同时属于多个类别的分类任务。

### 与多分类的区别

- **多分类（Multi-Class）**：每个样本只能属于一个类别
  - 例如：图片分类为"猫"或"狗"或"鸟"（只能选一个）
  - 输出：使用softmax，所有类别概率和为1
  
- **多标签分类（Multi-Label）**：每个样本可以同时属于多个类别
  - 例如：服装同时具有"黑色"、"纯色"、"棉质"等多个属性
  - 输出：使用sigmoid，每个类别独立预测，概率之间无关联

## 本服务的多标签分类

### 任务描述

给定一张服装图片，预测其具有的所有属性。模型会输出26个属性，每个属性都是一个二分类任务（有/无）。

### 26个属性列表

```python
[
    "black", "blue", "brown", "collar", "cyan",
    "gray", "green", "many_colors", "necktie", "pattern_floral",
    "pattern_graphics", "pattern_plaid", "pattern_solid", "pattern_spot", 
    "pattern_stripe", "pink", "purple", "red", "scarf", "skin_exposure",
    "white", "yellow", "denim", "knitted", "leather", "cotton"
]
```

### 分类流程

1. **输入图片** → 经过CNN特征提取
2. **特征增强** → 通过GAT+GCN图神经网络
3. **输出logits** → 26个属性的原始分数
4. **Sigmoid激活** → 转换为概率值（0-1之间）
5. **阈值分类** → 使用阈值将概率转换为0/1分类

### 输出说明

```json
{
  "attributes": {
    "black": 0.95,        // 置信度（概率值）
    "blue": 0.12,
    ...
  },
  "classifications": {
    "black": 1,           // 分类结果（0或1）
    "blue": 0,
    ...
  },
  "positive_attributes": [  // 预测为正的属性（按置信度排序）
    {"attribute": "black", "confidence": 0.95},
    {"attribute": "pattern_solid", "confidence": 0.89},
    ...
  ],
  "top_k_attributes": [    // Top-5置信度最高的属性
    {"attribute": "black", "confidence": 0.95, "classification": 1},
    ...
  ],
  "statistics": {
    "total_attributes": 26,
    "positive_count": 5,   // 预测为正的属性数量
    "threshold": 0.5       // 使用的阈值
  }
}
```

## 阈值调节

### 阈值的作用

阈值（threshold）用于将概率值转换为二分类结果：
- `概率 >= 阈值` → 分类为1（正类，该属性存在）
- `概率 < 阈值` → 分类为0（负类，该属性不存在）

### 选择合适的阈值

不同的阈值会影响分类结果：

| 阈值 | 效果 | 适用场景 |
|-----|------|---------|
| 0.3 | 更宽松，识别出更多属性 | 希望召回率高，不遗漏属性 |
| 0.5 | 平衡（默认） | 一般使用场景 |
| 0.7 | 更严格，只保留高置信度属性 | 希望精确率高，减少误判 |

### 示例对比

假设模型输出某属性的置信度为0.55：

```python
# 阈值=0.3
classification = 1  # 0.55 >= 0.3，判定为有该属性

# 阈值=0.5（默认）
classification = 1  # 0.55 >= 0.5，判定为有该属性

# 阈值=0.7
classification = 0  # 0.55 < 0.7，判定为无该属性
```

## API使用示例

### Python客户端

```python
import requests

# 1. 使用默认阈值0.5
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
result = response.json()

# 查看分类结果
print(f"预测为正的属性数量: {result['predictions']['statistics']['positive_count']}")
for attr in result['predictions']['positive_attributes']:
    print(f"  - {attr['attribute']}: {attr['confidence']:.2%}")

# 2. 使用自定义阈值0.6
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict?threshold=0.6',
        files={'file': f}
    )
result = response.json()
```

### cURL示例

```bash
# 默认阈值
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  | jq '.predictions.positive_attributes'

# 自定义阈值
curl -X POST "http://localhost:8000/predict?threshold=0.6" \
  -F "file=@image.jpg" \
  | jq '.predictions.positive_attributes'
```

## 评估指标

对于多标签分类任务，常用的评估指标包括：

### 1. 精确率（Precision）
```
Precision = TP / (TP + FP)
```
预测为正的样本中，真正为正的比例

### 2. 召回率（Recall）
```
Recall = TP / (TP + FN)
```
所有真正为正的样本中，被正确预测为正的比例

### 3. F1分数（F1-Score）
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
精确率和召回率的调和平均

### 4. Hamming Loss
```
Hamming Loss = (FP + FN) / (Total Labels)
```
预测错误的标签比例

## 最佳实践

### 1. 根据应用场景选择阈值

```python
# 电商推荐场景：希望多召回一些属性
threshold = 0.4

# 精准搜索场景：希望结果准确
threshold = 0.6

# 一般场景：使用默认值
threshold = 0.5
```

### 2. 批量处理优化

```python
# 批量处理多张图片时，使用统一阈值
images = ['img1.jpg', 'img2.jpg', 'img3.jpg']
files = [('files', open(img, 'rb')) for img in images]

response = requests.post(
    'http://localhost:8000/predict_batch?threshold=0.5',
    files=files
)

# 关闭文件
for _, f in files:
    f.close()
```

### 3. 分析置信度分布

```python
# 获取所有属性的置信度
confidences = result['predictions']['attributes']

# 分析分布
import numpy as np
values = list(confidences.values())
print(f"平均置信度: {np.mean(values):.2%}")
print(f"最大置信度: {np.max(values):.2%}")
print(f"最小置信度: {np.min(values):.2%}")
```

### 4. 使用正属性列表

```python
# 直接获取预测为正的属性
positive_attrs = result['predictions']['positive_attributes']

# 按置信度排序（已排序）
print("置信度最高的3个属性：")
for attr in positive_attrs[:3]:
    print(f"  {attr['attribute']}: {attr['confidence']:.2%}")
```

## 故障排查

### 1. 所有属性都被分类为0

可能原因：
- 阈值设置过高
- 图片质量问题
- 模型未正确加载

解决方案：
```python
# 降低阈值
threshold = 0.3

# 检查置信度分布
print(result['predictions']['attributes'])
```

### 2. 所有属性都被分类为1

可能原因：
- 阈值设置过低
- 模型训练问题

解决方案：
```python
# 提高阈值
threshold = 0.7

# 检查Top-K属性
print(result['predictions']['top_k_attributes'])
```

### 3. 结果不稳定

可能原因：
- 图片预处理问题
- 模型在边界情况

解决方案：
```python
# 使用多次预测取平均（如果需要）
# 或调整阈值范围
thresholds = [0.4, 0.5, 0.6]
results = []
for t in thresholds:
    result = predict(image, threshold=t)
    results.append(result)
```

## 参考资料

- [Multi-Label Classification](https://en.wikipedia.org/wiki/Multi-label_classification)
- [Evaluation Metrics for Multi-Label Classification](https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics)
- [Deep Learning for Multi-Label Classification](https://arxiv.org/abs/1502.05988)
