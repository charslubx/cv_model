# 论文实验完整指南

本指南说明如何运行论文第四章所需的全部实验，包括模块级消融实验、λ超参数实验和SOTA对比实验。

## 实验架构

本实验框架包含三大类实验，与论文第四章的实验设计一一对应：

### 1. 模块级消融实验（Ablation Study）

用于验证模型各个模块的有效性，包括：

| 模型配置 | 说明 |
|---------|------|
| **Baseline-CNN** | MSFE-FPN + FC，无图结构、无门控、无权重预测器 |
| **+Graph-GCN** | 加入静态kNN图 + GCN，不使用GAT |
| **+Graph-GAT** | 加入动态自适应图 + GAT，但不融合 |
| **+Fusion** | CNN + Graph 双分支门控融合，不加权重预测器 |
| **Full-AdaGAT** | 完整版 AdaGAT：融合 + 类别权重预测器 |

**实验目的**：证明每个模块（图结构、GAT、融合机制、权重预测器）对模型性能的贡献。

### 2. λ 超参数实验（Lambda Parameter Study）

测试动态阈值公式 `τ = μ + λσ` 中 λ 参数对模型性能的影响。

测试的 λ 值：`[0.0, 0.3, 0.5, 0.7, 1.0]`

**实验目的**：
- 为论文中 λ = 0.5 的选择提供实验依据
- 分析不同 λ 值对图构建的影响
- 生成可用于论文的实验结果和分析文本

**预期结论**：
> 实验表明，当 λ 取 0.5 时，在验证集上可获得最优的 F1 表现；
> 过小的 λ 导致图过于稠密，引入噪声邻居，
> 过大的 λ 又会使图过于稀疏，削弱邻域信息传递。
> 因此后续实验均固定 λ = 0.5。

### 3. SOTA 对比实验（State-of-the-Art Comparison）

将我们的方法与现有主流 CNN 方法进行公平对比。

对比的模型：
- **ResNet-50**：经典深度残差网络
- **ResNet-101**：更深的残差网络
- **EfficientNet-B0**：高效神经网络
- **DenseNet-121**：密集连接网络
- **ViT-B-16**：Vision Transformer
- **Our-AdaGAT**：我们提出的方法

**实验目的**：证明我们的方法相对于现有 CNN 方法的优越性。

## 快速开始

### 前置条件

确保已安装所有依赖：

```bash
pip install torch torchvision torch-geometric
pip install pandas matplotlib seaborn tqdm
pip install scikit-learn pillow
```

### 运行完整实验套件

```bash
python comprehensive_experiments.py
```

这将自动运行所有三类实验，并生成完整的实验报告。

**注意**：完整实验可能需要较长时间（取决于数据集大小和硬件配置）。

### 分别运行各类实验

如果只想运行特定类型的实验，可以修改 `comprehensive_experiments.py` 中的参数：

```python
# 在 main() 函数中修改
results = runner.run_all_experiments(
    run_ablation=True,   # 是否运行消融实验
    run_lambda=True,     # 是否运行λ实验
    run_sota=True,       # 是否运行SOTA对比
    epochs=30,           # 训练轮数
    learning_rate=3e-4   # 学习率
)
```

### 只运行 λ 超参数实验

如果只需要运行 λ 实验（推荐先运行这个，用时较短）：

```python
from lambda_experiment import LambdaExperiment
from dataset import create_data_loaders
from experiment_config import EXPERIMENT_CONFIG
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)

lambda_exp = LambdaExperiment(train_loader, val_loader, device, num_classes=26)
results = lambda_exp.run_experiment(epochs=30, learning_rate=3e-4, save_dir='lambda_results')
```

## 实验结果

### 结果目录结构

实验完成后，结果将保存在 `paper_experiments_YYYYMMDD_HHMMSS/` 目录下：

```
paper_experiments_YYYYMMDD_HHMMSS/
├── 1_ablation_study/                    # 消融实验结果
│   ├── Baseline-CNN_best.pth           # 最佳模型权重
│   ├── Baseline-CNN_results.json       # 详细结果
│   ├── +Graph-GCN_best.pth
│   ├── +Graph-GCN_results.json
│   ├── ... (其他模型)
│   ├── ablation_summary.csv            # 汇总表格
│   └── ablation_comparison.png         # 可视化对比图
│
├── 2_lambda_study/                      # λ实验结果
│   ├── best_model_lambda_0.0.pth       # 各λ值的最佳模型
│   ├── lambda_0.0_results.json         # 详细结果
│   ├── ... (其他λ值)
│   ├── lambda_experiment_summary.csv   # 汇总表格
│   ├── best_lambda.json                # 最佳λ值及结论
│   ├── lambda_vs_f1.png                # λ对F1的影响曲线
│   ├── training_curves_comparison.png  # 训练曲线对比
│   └── lambda_metrics_heatmap.png      # 指标热力图
│
├── 3_sota_comparison/                   # SOTA对比结果
│   ├── ResNet-50_best.pth
│   ├── ResNet-50_results.json
│   ├── ... (其他模型)
│   ├── Our-AdaGAT_best.pth             # 我们的模型
│   ├── sota_comparison_summary.csv     # 汇总表格
│   └── sota_comparison.png             # 可视化对比图
│
└── COMPREHENSIVE_REPORT.md              # 综合实验报告
```

### 关键文件说明

1. **ablation_summary.csv**：消融实验汇总表，可直接用于论文
2. **lambda_experiment_summary.csv**：λ实验汇总表
3. **best_lambda.json**：包含最佳λ值和论文可用的结论文本
4. **sota_comparison_summary.csv**：SOTA对比汇总表
5. **COMPREHENSIVE_REPORT.md**：完整的实验报告，包含所有表格和结论

## 论文写作指导

### 4.3 实验设置

可以使用以下表述：

> 本文在 DeepFashion 数据集的 Category and Attribute Prediction Benchmark 上进行实验。
> 训练集包含 X 张图像，验证集包含 Y 张图像。
> 所有模型使用 ResNet-50 作为特征提取主干，并在 ImageNet 上预训练。
> 训练超参数设置为：batch size = 32, learning rate = 3e-4, weight decay = 0.01。
> 使用 AdamW 优化器，学习率调度策略为 ReduceLROnPlateau。

### 4.4.1 消融实验

可以参考生成的 `ablation_summary.csv`，写作示例：

> 为验证所提方法各模块的有效性，本文设计了逐步添加模块的消融实验。
> 表 X 展示了消融实验结果。可以看出：
> 
> 1. Baseline-CNN 仅使用多尺度特征提取和简单分类头，F1 分数为 XX.XX%
> 2. 加入静态图结构后（+Graph-GCN），性能提升至 XX.XX%，说明图结构建模的有效性
> 3. 使用自适应GAT替换简单GCN（+Graph-GAT），F1进一步提升至 XX.XX%
> 4. 引入门控融合机制（+Fusion），将CNN和图分支有效结合，F1达到 XX.XX%
> 5. 完整的AdaGAT模型加入类别权重预测器，最终F1达到 XX.XX%
> 
> 消融实验证明了所提方法各个模块的必要性和有效性。

### 4.4.2 超参数分析

可以参考生成的 `best_lambda.json`，写作示例：

> 动态阈值公式 τ = μ + λσ 中的参数 λ 控制了图的稀疏程度。
> 本文测试了 λ ∈ {0.0, 0.3, 0.5, 0.7, 1.0} 对模型性能的影响，结果如图 X 所示。
> 
> 实验表明，当 λ 取 0.5 时，在验证集上可获得最优的 F1 表现（XX.XX%）。
> 过小的 λ（如0.0）导致图过于稠密，引入噪声邻居，反而降低模型性能；
> 过大的 λ（如1.0）又会使图过于稀疏，削弱邻域信息传递的效果。
> 因此后续实验均固定 λ = 0.5。

### 4.4.3 SOTA对比

可以参考生成的 `sota_comparison_summary.csv`，写作示例：

> 为验证所提方法的有效性，本文将 AdaGAT 与多种主流方法进行对比，结果如表 X 所示。
> 
> 可以看出，我们的 AdaGAT 方法在 F1 分数上达到 XX.XX%，
> 相比最优的 CNN 方法（XXX）提升了 X.XX 个百分点。
> 这证明了图结构建模和自适应注意力机制在服装属性识别任务中的优越性。

## 常见问题

### Q1: 实验运行时间过长怎么办？

A: 可以调整以下参数：
- 减少 `epochs`（如从30减到10-15）
- 减小 `batch_size`（注意可能影响性能）
- 只运行必要的实验（如先运行λ实验）

### Q2: 显存不足怎么办？

A: 
- 减小 `batch_size`
- 使用梯度累积
- 使用混合精度训练（需要修改代码添加 AMP）

### Q3: 如何修改测试的 λ 值范围？

A: 修改 `lambda_experiment.py` 中的 `self.lambda_values`：

```python
self.lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]  # 修改为你需要的值
```

### Q4: 如何添加其他 SOTA 模型进行对比？

A: 在 `comprehensive_experiments.py` 中的 `_run_sota_experiments` 方法中添加：

```python
sota_configs = [
    {'name': 'ResNet-50', 'backbone': 'resnet50'},
    # 添加你的模型配置
    {'name': 'Your-Model', 'backbone': 'your_backbone'},
]
```

## 代码文件说明

| 文件 | 说明 |
|------|------|
| `ablation_models.py` | 所有消融实验的模型变体定义 |
| `lambda_experiment.py` | λ 超参数实验框架 |
| `comprehensive_experiments.py` | 完整实验运行器（主入口） |
| `PAPER_EXPERIMENTS_GUIDE.md` | 本指南文件 |

## 技术支持

如果遇到问题，请检查：
1. 数据集路径是否正确配置
2. Python 环境和依赖是否完整安装
3. CUDA 和 PyTorch 版本是否兼容
4. 日志文件 `comprehensive_experiments.log` 中的错误信息

## 引用

如果本实验框架对您的研究有帮助，欢迎引用我们的工作。

---

**祝您的论文实验顺利完成！**

