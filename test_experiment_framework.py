"""
测试实验框架的快速验证脚本
在运行完整实验前，先用小规模数据测试框架是否正常工作
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import sys

# 导入我们的模型
from ablation_models import (
    BaselineCNN, 
    CNNWithStaticGraph, 
    CNNWithAdaptiveGAT, 
    CNNGraphFusion, 
    FullAdaGAT
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """虚拟数据集，用于快速测试"""
    
    def __init__(self, num_samples=100, num_classes=26):
        self.num_samples = num_samples
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机图像和标签
        image = torch.randn(3, 224, 224)
        labels = torch.randint(0, 2, (self.num_classes,)).float()
        
        return {
            'image': image,
            'attr_labels': labels
        }


def test_model(model, model_name, device, data_loader):
    """测试单个模型的前向传播"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试模型: {model_name}")
    logger.info(f"{'='*60}")
    
    model = model.to(device)
    model.eval()
    
    try:
        # 测试前向传播
        batch = next(iter(data_loader))
        images = batch['image'].to(device)
        labels = batch['attr_labels'].to(device)
        
        with torch.no_grad():
            outputs = model(images)
        
        # 检查输出
        assert 'attr_logits' in outputs, "输出缺少 attr_logits"
        assert outputs['attr_logits'].shape == labels.shape, "输出形状不匹配"
        
        logger.info(f"✓ 前向传播成功")
        logger.info(f"  输入形状: {images.shape}")
        logger.info(f"  输出形状: {outputs['attr_logits'].shape}")
        
        # 测试损失计算
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(outputs['attr_logits'], labels)
        logger.info(f"✓ 损失计算成功: {loss.item():.4f}")
        
        # 测试反向传播
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs['attr_logits'], labels)
        loss.backward()
        optimizer.step()
        
        logger.info(f"✓ 反向传播成功")
        logger.info(f"✓ {model_name} 测试通过！")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ {model_name} 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("\n" + "="*80)
    logger.info("开始测试实验框架")
    logger.info("="*80)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n使用设备: {device}")
    
    # 创建虚拟数据集
    logger.info("\n创建测试数据集...")
    dataset = DummyDataset(num_samples=32, num_classes=26)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    logger.info(f"数据集大小: {len(dataset)}")
    logger.info(f"Batch size: 8")
    
    # 定义要测试的模型
    models_to_test = [
        ('Baseline-CNN', BaselineCNN(num_classes=26)),
        ('+Graph-GCN', CNNWithStaticGraph(num_classes=26)),
        ('+Graph-GAT', CNNWithAdaptiveGAT(num_classes=26, lambda_threshold=0.5)),
        ('+Fusion', CNNGraphFusion(num_classes=26, lambda_threshold=0.5)),
        ('Full-AdaGAT', FullAdaGAT(num_classes=26, lambda_threshold=0.5))
    ]
    
    # 测试每个模型
    results = {}
    for model_name, model in models_to_test:
        success = test_model(model, model_name, device, data_loader)
        results[model_name] = success
    
    # 打印测试总结
    logger.info("\n" + "="*80)
    logger.info("测试总结")
    logger.info("="*80)
    
    all_passed = True
    for model_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        logger.info(f"{model_name:20s} - {status}")
        if not success:
            all_passed = False
    
    logger.info("\n" + "="*80)
    if all_passed:
        logger.info("✓ 所有模型测试通过！实验框架可以正常使用。")
        logger.info("="*80)
        logger.info("\n现在可以运行完整实验：")
        logger.info("  python comprehensive_experiments.py")
        return 0
    else:
        logger.error("✗ 部分模型测试失败，请检查错误信息。")
        logger.info("="*80)
        return 1


if __name__ == '__main__':
    sys.exit(main())

