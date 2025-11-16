"""
诊断F1分数过低的问题
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import create_data_loaders
from experiment_config import EXPERIMENT_CONFIG
from ablation_models import CNNWithAdaptiveGAT
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_quality():
    """检查数据质量"""
    logger.info("\n" + "="*80)
    logger.info("第1步：检查数据质量")
    logger.info("="*80)
    
    train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
    
    # 检查数据集大小
    logger.info(f"\n数据集大小:")
    logger.info(f"  训练集: {len(train_loader.dataset)} 样本")
    logger.info(f"  验证集: {len(val_loader.dataset)} 样本")
    logger.info(f"  Batch size: {train_loader.batch_size}")
    logger.info(f"  训练batches: {len(train_loader)}")
    logger.info(f"  验证batches: {len(val_loader)}")
    
    if len(train_loader.dataset) < 1000:
        logger.warning(f"⚠ 训练集样本太少！只有 {len(train_loader.dataset)} 个样本")
    
    # 检查标签分布
    logger.info(f"\n检查标签分布...")
    all_labels = []
    for i, batch in enumerate(train_loader):
        if i >= 10:  # 只检查前10个batch
            break
        labels = batch['attr_labels']
        all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0)
    logger.info(f"  检查的样本数: {len(all_labels)}")
    logger.info(f"  标签形状: {all_labels.shape}")
    logger.info(f"  标签范围: [{all_labels.min():.3f}, {all_labels.max():.3f}]")
    
    # 检查每个属性的正样本比例
    pos_ratios = all_labels.mean(dim=0)
    logger.info(f"\n每个属性的正样本比例:")
    for i, ratio in enumerate(pos_ratios):
        status = ""
        if ratio < 0.05:
            status = " ⚠ 极度稀疏"
        elif ratio > 0.95:
            status = " ⚠ 几乎全是正样本"
        logger.info(f"  属性 {i:2d}: {ratio:.3f}{status}")
    
    avg_pos_ratio = pos_ratios.mean().item()
    logger.info(f"\n平均正样本比例: {avg_pos_ratio:.3f}")
    
    if avg_pos_ratio < 0.1 or avg_pos_ratio > 0.9:
        logger.warning(f"⚠ 严重的类别不平衡！")
    
    return train_loader, val_loader, avg_pos_ratio


def check_model_initialization():
    """检查模型初始化"""
    logger.info("\n" + "="*80)
    logger.info("第2步：检查模型初始化")
    logger.info("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNWithAdaptiveGAT(
        num_classes=26,
        cnn_type='resnet50',
        weights='IMAGENET1K_V1',  # 关键：检查是否正确加载
        lambda_threshold=0.5
    ).to(device)
    
    # 检查是否有预训练权重
    has_pretrained = False
    try:
        # 检查第一个卷积层的权重
        first_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is not None:
            weight_mean = first_conv.weight.data.mean().item()
            weight_std = first_conv.weight.data.std().item()
            logger.info(f"\n第一个卷积层权重统计:")
            logger.info(f"  均值: {weight_mean:.6f}")
            logger.info(f"  标准差: {weight_std:.6f}")
            
            # ImageNet预训练权重通常不会是全0或者方差太小
            if abs(weight_mean) < 0.001 and weight_std < 0.01:
                logger.warning(f"⚠ 权重看起来像随机初始化，可能没有加载预训练权重")
            else:
                logger.info(f"✓ 权重看起来已加载预训练")
                has_pretrained = True
    except Exception as e:
        logger.error(f"检查权重时出错: {e}")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n模型参数:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")
    
    return model, has_pretrained


def check_forward_pass(model, train_loader):
    """检查前向传播"""
    logger.info("\n" + "="*80)
    logger.info("第3步：检查前向传播")
    logger.info("="*80)
    
    device = next(model.parameters()).device
    model.eval()
    
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    labels = batch['attr_labels'].to(device)
    
    logger.info(f"\n输入:")
    logger.info(f"  图像形状: {images.shape}")
    logger.info(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
    logger.info(f"  标签形状: {labels.shape}")
    logger.info(f"  标签均值: {labels.mean():.3f}")
    
    with torch.no_grad():
        outputs = model(images)
    
    logits = outputs['attr_logits']
    probs = torch.sigmoid(logits)
    
    logger.info(f"\n输出:")
    logger.info(f"  Logits形状: {logits.shape}")
    logger.info(f"  Logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
    logger.info(f"  Logits均值: {logits.mean():.3f}")
    logger.info(f"  Logits标准差: {logits.std():.3f}")
    
    logger.info(f"\n预测概率:")
    logger.info(f"  概率范围: [{probs.min():.3f}, {probs.max():.3f}]")
    logger.info(f"  概率均值: {probs.mean():.3f}")
    logger.info(f"  预测为正的比例: {(probs > 0.5).float().mean():.3f}")
    logger.info(f"  真实正样本比例: {labels.mean():.3f}")
    
    # 检查是否所有预测都相同
    if probs.std() < 0.01:
        logger.warning(f"⚠ 模型预测几乎没有变化，可能训练有问题")
    
    # 计算简单准确率
    binary_preds = (probs > 0.5).float()
    accuracy = (binary_preds == labels).float().mean()
    logger.info(f"\nBatch准确率: {accuracy:.3f}")
    
    if accuracy < 0.3:
        logger.warning(f"⚠ 准确率太低！")


def check_loss_and_gradients(model, train_loader):
    """检查损失和梯度"""
    logger.info("\n" + "="*80)
    logger.info("第4步：检查损失和梯度")
    logger.info("="*80)
    
    device = next(model.parameters()).device
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    batch = next(iter(train_loader))
    images = batch['image'].to(device)
    labels = batch['attr_labels'].to(device)
    
    # 前向传播
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs['attr_logits'], labels)
    
    logger.info(f"\n损失:")
    logger.info(f"  Loss值: {loss.item():.4f}")
    
    # 理论上BCE loss应该在0-1之间，如果太高说明有问题
    if loss.item() > 1.0:
        logger.warning(f"⚠ 损失值异常高！")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    total_norm = 0
    max_grad = 0
    min_grad = float('inf')
    grad_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            max_grad = max(max_grad, param.grad.abs().max().item())
            min_grad = min(min_grad, param.grad.abs().min().item())
            grad_count += 1
    
    total_norm = total_norm ** 0.5
    
    logger.info(f"\n梯度:")
    logger.info(f"  总梯度范数: {total_norm:.6f}")
    logger.info(f"  最大梯度: {max_grad:.6f}")
    logger.info(f"  最小梯度: {min_grad:.6f}")
    logger.info(f"  有梯度的参数数: {grad_count}")
    
    if total_norm < 1e-6:
        logger.warning(f"⚠ 梯度几乎为0，模型可能无法学习")
    elif total_norm > 100:
        logger.warning(f"⚠ 梯度爆炸！")


def main():
    """主诊断函数"""
    logger.info("\n" + "="*80)
    logger.info("开始诊断F1分数过低的问题")
    logger.info("="*80)
    
    try:
        # 1. 检查数据
        train_loader, val_loader, avg_pos_ratio = check_data_quality()
        
        # 2. 检查模型
        model, has_pretrained = check_model_initialization()
        
        # 3. 检查前向传播
        check_forward_pass(model, train_loader)
        
        # 4. 检查损失和梯度
        check_loss_and_gradients(model, train_loader)
        
        # 综合诊断
        logger.info("\n" + "="*80)
        logger.info("诊断总结")
        logger.info("="*80)
        
        issues = []
        suggestions = []
        
        if len(train_loader.dataset) < 1000:
            issues.append("❌ 训练集样本太少")
            suggestions.append("检查数据集路径是否正确")
        
        if avg_pos_ratio < 0.1 or avg_pos_ratio > 0.9:
            issues.append("❌ 严重的类别不平衡")
            suggestions.append("使用加权损失或Focal Loss")
        
        if not has_pretrained:
            issues.append("❌ 可能没有正确加载预训练权重")
            suggestions.append("检查weights参数是否正确")
        
        if issues:
            logger.info("\n发现的问题:")
            for issue in issues:
                logger.info(f"  {issue}")
            
            logger.info("\n建议的解决方案:")
            for i, suggestion in enumerate(suggestions, 1):
                logger.info(f"  {i}. {suggestion}")
        else:
            logger.info("\n✓ 未发现明显问题")
            logger.info("  可能需要:")
            logger.info("  1. 增加训练轮数（当前可能只训练了2个epoch）")
            logger.info("  2. 调整学习率")
            logger.info("  3. 检查是否使用了正确的损失函数")
        
        logger.info("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"\n诊断过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

