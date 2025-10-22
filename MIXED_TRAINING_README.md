# 混合数据集训练指南

本文档介绍如何使用DeepFashion和TextileNet数据集进行混合训练。

## 概述

混合训练系统结合了三个数据集：
1. **DeepFashion**: 服装属性分类和分析
2. **Fabric**: 面料纹理分类（20个类别）
3. **Fiber**: 纤维材质分类（32个类别）

## 主要改进

### 1. 模型保存方式
- **修改前**: 仅保存模型权重 (`state_dict`)
- **修改后**: 保存完整模型，便于直接加载使用

### 2. 新增数据集支持
- **TextileNetDataset**: 支持fabric和fiber数据集加载
- **MixedDataset**: 混合多个数据集，支持多种混合策略
- **MixedDatasetTrainer**: 支持多任务混合训练

### 3. 模型架构增强
- 新增纹理分类头 (`fabric_head`, `fiber_head`, `textile_head`)
- 支持多任务学习（属性分类 + 纹理分类 + 分割）
- 动态损失加权机制

## 数据集结构

```
/home/cv_model/
├── DeepFashion/                    # DeepFashion数据集
│   └── Category and Attribute Prediction Benchmark/
│       ├── Img/img/               # 图像文件
│       └── Anno_fine/             # 标注文件
├── fabric/                        # Fabric纹理数据集
│   ├── train/
│   │   ├── canvas/               # 帆布
│   │   ├── denim/                # 牛仔布
│   │   ├── lace/                 # 蕾丝
│   │   └── ...                   # 其他面料类别
│   └── test/                     # 测试集（可选）
└── fiber/                        # Fiber纤维数据集
    ├── train/
    │   ├── cotton/               # 棉花
    │   ├── silk/                 # 丝绸
    │   ├── wool/                 # 羊毛
    │   └── ...                   # 其他纤维类别
    └── test/                     # 测试集（可选）
```

## 使用方法

### 1. 基本训练

```python
# 运行混合训练脚本
python run_mixed_training.py
```

### 2. 自定义配置

```python
from training import MixedDataset, MixedDatasetTrainer
from base_model import FullModel

# 创建混合数据集
mixed_dataset = MixedDataset(
    deepfashion_dataset=deepfashion_train,
    fabric_dataset=fabric_train,
    fiber_dataset=fiber_train,
    mixing_strategy='balanced',  # 'balanced', 'weighted', 'sequential'
    deepfashion_weight=0.5,
    fabric_weight=0.25,
    fiber_weight=0.25
)

# 创建增强模型
model = FullModel(
    num_classes=26,                    # DeepFashion属性数
    enable_textile_classification=True, # 启用纹理分类
    num_fabric_classes=20,             # Fabric类别数
    num_fiber_classes=32,              # Fiber类别数
    gat_dims=[1024, 512],
    gat_heads=8
)

# 创建训练器
trainer = MixedDatasetTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    enable_textile_classification=True
)

# 开始训练
trainer.train(epochs=50, save_dir="checkpoints")
```

## 混合策略

### 1. Balanced (平衡策略)
- 根据权重随机采样各数据集
- 适合数据集大小差异较大的情况

### 2. Weighted (加权策略)
- 按权重比例分配样本数量
- 适合需要精确控制各数据集比例的情况

### 3. Sequential (顺序策略)
- 依次使用各数据集的所有样本
- 适合需要遍历所有数据的情况

## 模型输出

模型现在支持多种输出：

```python
outputs = model(images)

# DeepFashion属性分类
attr_logits = outputs['attr_logits']        # [batch_size, 26]

# 纹理分类
fabric_logits = outputs['fabric_logits']    # [batch_size, num_fabric_classes]
fiber_logits = outputs['fiber_logits']      # [batch_size, num_fiber_classes]
textile_logits = outputs['textile_logits']  # [batch_size, max_classes]

# 分割（如果启用）
seg_logits = outputs['seg_logits']          # [batch_size, 1, H, W]

# 其他辅助输出
class_weights = outputs['class_weights']    # [batch_size, 26]
```

## 损失函数

系统使用多任务损失函数：

1. **属性分类损失**: Focal Loss（处理类别不平衡）
2. **纹理分类损失**: Cross Entropy Loss
3. **分割损失**: BCE with Logits Loss
4. **多任务损失**: 动态权重平衡各任务

## 训练监控

训练过程中会输出以下指标：

```
Epoch 1/50
训练指标:
- total_loss: 2.1234
- attr_loss: 1.5678
- textile_loss: 0.5556
- seg_loss: 0.0000

验证指标:
- precision: 0.7234
- recall: 0.6789
- f1: 0.7000
- threshold: 0.5000
```

## 模型保存和加载

### 保存模型
```python
# 训练器会自动保存完整模型
# 保存路径: {save_dir}/best_model.pth
```

### 加载模型
```python
import torch

# 加载完整模型
model = torch.load('checkpoints/best_model.pth')
model.eval()

# 进行推理
with torch.no_grad():
    outputs = model(images)
```

## 注意事项

1. **内存使用**: 混合训练会增加内存使用，建议使用GPU训练
2. **数据平衡**: 注意调整各数据集的权重以获得最佳效果
3. **收敛速度**: 多任务学习可能需要更多训练轮次
4. **验证策略**: 建议分别验证各任务的性能

## 故障排除

### 常见问题

1. **数据集路径错误**
   - 检查数据集路径是否正确
   - 确保图像文件存在且可读

2. **内存不足**
   - 减小batch_size
   - 使用更少的worker进程

3. **收敛困难**
   - 调整学习率
   - 修改损失权重
   - 增加训练轮次

### 调试模式

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查数据集信息
dataset_info = mixed_dataset.get_dataset_info()
print(dataset_info)
```

## 性能优化建议

1. **数据加载优化**
   - 使用多进程数据加载 (`num_workers > 0`)
   - 启用内存固定 (`pin_memory=True`)

2. **训练优化**
   - 使用混合精度训练 (AMP)
   - 启用EMA模型
   - 使用学习率调度器

3. **模型优化**
   - 调整GAT层数和头数
   - 优化特征融合策略
   - 使用更强的backbone网络

## 扩展功能

系统支持以下扩展：

1. **新数据集集成**: 继承`Dataset`类实现新的数据集
2. **自定义损失函数**: 修改训练器中的损失计算
3. **模型架构改进**: 在`FullModel`中添加新的分支
4. **评估指标扩展**: 在验证函数中添加新的指标

通过这个混合训练系统，您可以充分利用多个数据集的优势，训练出更强大的服装分析模型。
