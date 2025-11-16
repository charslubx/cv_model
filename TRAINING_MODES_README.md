# 训练模式说明

本项目提供两种训练模式，用于不同的任务场景。

## 模式一：DeepFashion单独训练（推荐用于论文实验）

### 用途
- 训练FullAdaGAT模型进行服装属性识别
- 用于论文实验（消融实验、SOTA对比等）
- 生成可用于`comprehensive_experiments.py`的预训练模型

### 使用方法

```bash
# 基本用法
python train_deepfashion.py

# 自定义参数
python train_deepfashion.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 3e-4 \
    --save_dir deepfashion_checkpoints \
    --lambda_threshold 0.5
```

### 参数说明
- `--epochs`: 训练轮数（默认30）
- `--batch_size`: 批次大小（默认32）
- `--lr`: 学习率（默认3e-4）
- `--save_dir`: 模型保存目录（默认deepfashion_checkpoints）
- `--lambda_threshold`: FullAdaGAT的lambda阈值（默认0.5）

### 输出
- 训练好的模型保存为state_dict格式：`{save_dir}/best_model.pth`
- 该模型可直接用于`comprehensive_experiments.py`的消融实验和SOTA对比

### 特点
- 只使用DeepFashion数据集
- 使用FullAdaGAT模型架构
- 保存为state_dict格式，便于模型加载和复用
- 自动计算F1、Precision、Recall等指标
- 支持早停和学习率调度

## 模式二：混合数据集训练

### 用途
- 同时训练多个任务（属性识别、面料分类、纤维分类）
- 使用DeepFashion + Fabric + Fiber数据集
- 生成多任务FullModel模型

### 使用方法

```bash
# 使用混合训练模式
python training.py --mode mixed

# 自定义参数
python training.py \
    --mode mixed \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir smart_mixed_checkpoints
```

### 参数说明
- `--mode`: 训练模式，设置为`mixed`
- `--epochs`: 训练轮数（默认30）
- `--batch_size`: 批次大小（默认32）
- `--lr`: 学习率（默认3e-4）
- `--save_dir`: 模型保存目录（默认smart_mixed_checkpoints）

### 输出
- 训练好的模型保存为完整模型：`{save_dir}/best_model.pth`
- 该模型包含多个输出头（attr_head, fabric_head, fiber_head）

### 特点
- 智能检测可用数据集
- 自动调整数据集权重
- 使用FullModel模型架构
- 支持多任务学习
- 保存完整模型对象

## 模型架构对比

| 特性 | DeepFashion单独训练 | 混合数据集训练 |
|------|-------------------|--------------|
| 模型类 | FullAdaGAT | FullModel |
| 任务 | 单任务（属性识别） | 多任务（属性+面料+纤维） |
| 数据集 | DeepFashion | DeepFashion + Fabric + Fiber |
| 保存格式 | state_dict | 完整模型 |
| 用途 | 论文实验 | 实际应用 |

## 论文实验工作流

如果要进行论文实验，推荐以下工作流：

1. **训练FullAdaGAT模型**
   ```bash
   python train_deepfashion.py --epochs 30 --save_dir deepfashion_checkpoints
   ```

2. **运行论文实验**
   ```bash
   python comprehensive_experiments.py
   ```
   
   实验框架会自动：
   - 在消融实验中加载`deepfashion_checkpoints/best_model.pth`
   - 在SOTA对比中复用该预训练模型
   - 训练其他基线模型进行对比

3. **查看结果**
   - 实验结果保存在`paper_experiments_{timestamp}/`目录
   - 包含消融实验、lambda实验、SOTA对比的完整结果

## 注意事项

1. **数据集路径**：确保DeepFashion数据集位于`/home/cv_model/DeepFashion`

2. **GPU内存**：如果显存不足，可以减小batch_size：
   ```bash
   python train_deepfashion.py --batch_size 16
   ```

3. **模型兼容性**：
   - DeepFashion单独训练生成的模型只能用于属性识别任务
   - 混合训练生成的模型支持多任务，但不能直接用于论文实验框架

4. **预训练模型**：
   - 论文实验需要使用DeepFashion单独训练的模型
   - 确保模型文件路径正确，否则会自动重新训练

## 故障排除

### 问题1：模型加载失败
```
错误: 模型类型不匹配: FullModel，需要FullAdaGAT
```
**解决方案**：使用`train_deepfashion.py`重新训练，而不是使用混合训练的模型。

### 问题2：数据集找不到
```
错误: DeepFashion数据集路径不存在
```
**解决方案**：检查数据集路径是否正确，确保数据集已正确解压。

### 问题3：显存不足
```
错误: CUDA out of memory
```
**解决方案**：减小batch_size或使用梯度累积。

