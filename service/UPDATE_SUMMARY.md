# 服务更新说明 - 多标签分类

## 更新日期
2025-10-17

## 更新原因
根据base_model.py的实际实现，模型执行的是**多标签分类**任务，而非简单的预测任务。每张图片可以同时具有多个属性。

## 主要更改

### 1. 模型推理逻辑 (`model_loader.py`)

#### 更新前（错误）
```python
# 只返回置信度，没有分类结果
attr_probs = torch.sigmoid(attr_logits).cpu().numpy()[0]
predictions = {
    "attributes": {...},  # 只有置信度
    "top_k_attributes": [...]
}
```

#### 更新后（正确）
```python
# 返回置信度和分类结果
attr_probs = torch.sigmoid(attr_logits).cpu().numpy()[0]
attr_classes = (attr_probs >= threshold).astype(int)  # 二值化分类

predictions = {
    "attributes": {...},              # 置信度（0-1）
    "classifications": {...},          # 分类结果（0或1）
    "positive_attributes": [...],      # 预测为正的属性
    "top_k_attributes": [...],
    "statistics": {
        "total_attributes": 26,
        "positive_count": 5,           # 预测为正的数量
        "threshold": 0.5               # 使用的阈值
    }
}
```

### 2. API接口更新 (`app.py`)

#### 新增threshold参数
```python
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.5  # 新增：分类阈值
):
    ...
```

#### 调用示例
```bash
# 使用默认阈值0.5
curl -X POST "http://localhost:8000/predict" -F "file=@image.jpg"

# 使用自定义阈值0.6
curl -X POST "http://localhost:8000/predict?threshold=0.6" -F "file=@image.jpg"
```

### 3. 输出格式对比

#### 更新前
```json
{
  "success": true,
  "filename": "image.jpg",
  "predictions": {
    "attributes": {
      "black": 0.95,
      "blue": 0.12,
      "pattern_solid": 0.89
    },
    "top_k_attributes": [
      {"attribute": "black", "confidence": 0.95}
    ]
  }
}
```

#### 更新后（新增字段）
```json
{
  "success": true,
  "filename": "image.jpg",
  "predictions": {
    "attributes": {
      "black": 0.95,
      "blue": 0.12,
      "pattern_solid": 0.89
    },
    "classifications": {              // 新增：分类结果
      "black": 1,
      "blue": 0,
      "pattern_solid": 1
    },
    "positive_attributes": [          // 新增：预测为正的属性
      {"attribute": "black", "confidence": 0.95},
      {"attribute": "pattern_solid", "confidence": 0.89}
    ],
    "top_k_attributes": [
      {
        "attribute": "black",
        "confidence": 0.95,
        "classification": 1           // 新增：包含分类结果
      }
    ],
    "statistics": {                   // 新增：统计信息
      "total_attributes": 26,
      "positive_count": 2,
      "threshold": 0.5
    }
  }
}
```

### 4. 新增文档

#### CLASSIFICATION_GUIDE.md
详细的多标签分类指南，包含：
- 多标签分类概念说明
- 与多分类的区别
- 阈值选择指南
- 使用示例
- 评估指标
- 最佳实践
- 故障排查

### 5. 客户端示例更新 (`client_example.py`)

```python
# 旧版本
result = client.predict(image_path)
print(result['predictions']['top_k_attributes'])

# 新版本
# 使用默认阈值
result = client.predict(image_path)
print(f"预测为正的属性: {result['predictions']['positive_attributes']}")

# 使用自定义阈值
result = client.predict(image_path, threshold=0.7)
print(f"高阈值下预测为正: {result['predictions']['statistics']['positive_count']}")
```

## 向后兼容性

### ✅ 完全兼容
所有旧的字段仍然保留：
- `attributes`: 置信度字典
- `top_k_attributes`: Top-K属性列表

### ✨ 新增字段
- `classifications`: 分类结果字典（0或1）
- `positive_attributes`: 预测为正的属性列表
- `statistics`: 统计信息
- `threshold`: API参数，控制分类阈值

### 📝 字段变化
`top_k_attributes`中的每个项目新增`classification`字段

## 使用建议

### 1. 对于新用户
推荐使用新的字段：
```python
# 直接获取预测为正的属性
positive_attrs = result['predictions']['positive_attributes']
for attr in positive_attrs:
    print(f"{attr['attribute']}: {attr['confidence']:.2%}")
```

### 2. 对于已有代码
无需修改，原有代码继续工作：
```python
# 仍然可以使用
attrs = result['predictions']['attributes']
top_k = result['predictions']['top_k_attributes']
```

### 3. 阈值调优
根据应用场景选择合适的阈值：

```python
# 召回优先（识别更多属性）
result = client.predict(image, threshold=0.4)

# 精确优先（减少误判）
result = client.predict(image, threshold=0.6)

# 平衡（默认）
result = client.predict(image, threshold=0.5)
```

## 技术细节

### 模型输出流程

```
输入图片
  ↓
CNN特征提取（ResNet50）
  ↓
GAT图注意力网络
  ↓
GCN图卷积分类
  ↓
输出logits [batch_size, 26]
  ↓
Sigmoid激活 → 概率 [0-1]
  ↓
阈值分类 → 二值结果 [0或1]
```

### 损失函数
模型使用FocalLoss进行训练，这是针对多标签分类的损失函数：

```python
class FocalLoss(nn.Module):
    def forward(self, inputs, targets):
        # inputs: logits
        # targets: 0或1的多标签
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        ...
```

### 评估指标
多标签分类常用指标：
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (P * R) / (P + R)
- **Hamming Loss**: 错误预测的标签比例

## 示例对比

### 场景：分析一件黑色纯色棉质衬衫

#### 模型输出（置信度）
```python
{
    "black": 0.95,
    "pattern_solid": 0.89,
    "cotton": 0.78,
    "collar": 0.68,
    "blue": 0.12,
    "pattern_stripe": 0.08,
    ...
}
```

#### 使用阈值0.5的分类结果
```python
{
    "black": 1,          # 0.95 >= 0.5 ✓
    "pattern_solid": 1,  # 0.89 >= 0.5 ✓
    "cotton": 1,         # 0.78 >= 0.5 ✓
    "collar": 1,         # 0.68 >= 0.5 ✓
    "blue": 0,           # 0.12 < 0.5 ✗
    "pattern_stripe": 0, # 0.08 < 0.5 ✗
    ...
}
```

#### 预测为正的属性
```python
[
    {"attribute": "black", "confidence": 0.95},
    {"attribute": "pattern_solid", "confidence": 0.89},
    {"attribute": "cotton", "confidence": 0.78},
    {"attribute": "collar", "confidence": 0.68}
]
```

结果：模型正确识别出4个属性

## 迁移指南

### 如果你之前使用top_k_attributes
```python
# 之前的代码
for attr in result['predictions']['top_k_attributes']:
    print(f"{attr['attribute']}: {attr['confidence']}")

# 建议改为
for attr in result['predictions']['positive_attributes']:
    print(f"{attr['attribute']}: {attr['confidence']}")
    # 这样只显示实际预测为正的属性
```

### 如果你需要严格的分类结果
```python
# 新代码
classifications = result['predictions']['classifications']
for attr_name, cls in classifications.items():
    if cls == 1:
        print(f"服装具有属性: {attr_name}")
```

## 性能影响

### ⚡ 无性能影响
新增的分类逻辑在CPU上执行，开销极小：
```python
# 阈值分类只是简单的比较操作
attr_classes = (attr_probs >= threshold).astype(int)
```

### 📊 内存影响
新增字段增加的内存可忽略不计（每张图片约几KB）

## 测试验证

运行测试脚本验证更新：
```bash
python service/check_setup.py
```

## 问题反馈

如有问题，请查看：
1. `service/README.md` - 服务文档
2. `service/CLASSIFICATION_GUIDE.md` - 分类指南
3. `service/client_example.py` - 使用示例

---

**更新版本**: v1.1.0  
**提交哈希**: 801cb24  
**更新时间**: 2025-10-17
