"""
实验调试脚本
用于诊断F1分数过低的问题
"""

import torch
import torch.nn as nn
from dataset import create_data_loaders
from experiment_config import EXPERIMENT_CONFIG
from ablation_models import CNNWithAdaptiveGAT
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_distribution(data_loader, num_batches=5):
    """检查数据分布"""
    logger.info("\n" + "="*80)
    logger.info("检查数据分布")
    logger.info("="*80)
    
    all_labels = []
    all_images = []
    
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        
        images = batch['image']
        labels = batch['attr_labels']
        
        all_images.append(images)
        all_labels.append(labels)
        
        logger.info(f"\nBatch {i+1}:")
        logger.info(f"  图像形状: {images.shape}")
        logger.info(f"  标签形状: {labels.shape}")
        logger.info(f"  图像范围: [{images.min():.3f}, {images.max():.3f}]")
        logger.info(f"  标签范围: [{labels.min():.3f}, {labels.max():.3f}]")
        logger.info(f"  正样本比例: {labels.mean():.3f}")
    
    # 汇总统计
    all_labels = torch.cat(all_labels, dim=0)
    all_images = torch.cat(all_images, dim=0)
    
    logger.info("\n" + "="*80)
    logger.info("数据汇总统计")
    logger.info("="*80)
    logger.info(f"总样本数: {len(all_labels)}")
    logger.info(f"图像形状: {all_images.shape}")
    logger.info(f"标签形状: {all_labels.shape}")
    logger.info(f"图像均值: {all_images.mean():.3f}")
    logger.info(f"图像标准差: {all_images.std():.3f}")
    logger.info(f"总体正样本比例: {all_labels.mean():.3f}")
    
    # 每个属性的统计
    logger.info("\n每个属性的正样本比例:")
    for i in range(all_labels.shape[1]):
        pos_ratio = all_labels[:, i].mean().item()
        logger.info(f"  属性 {i:2d}: {pos_ratio:.3f}")
    
    return all_labels.mean().item()


def check_model_output(model, data_loader, device):
    """检查模型输出"""
    logger.info("\n" + "="*80)
    logger.info("检查模型输出")
    logger.info("="*80)
    
    model.eval()
    batch = next(iter(data_loader))
    images = batch['image'].to(device)
    labels = batch['attr_labels'].to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    logits = outputs['attr_logits']
    probs = torch.sigmoid(logits)
    
    logger.info(f"\n输入形状: {images.shape}")
    logger.info(f"标签形状: {labels.shape}")
    logger.info(f"输出logits形状: {logits.shape}")
    logger.info(f"Logits范围: [{logits.min():.3f}, {logits.max():.3f}]")
    logger.info(f"Logits均值: {logits.mean():.3f}")
    logger.info(f"Logits标准差: {logits.std():.3f}")
    logger.info(f"概率范围: [{probs.min():.3f}, {probs.max():.3f}]")
    logger.info(f"概率均值: {probs.mean():.3f}")
    logger.info(f"预测正样本比例: {(probs > 0.5).float().mean():.3f}")
    logger.info(f"真实正样本比例: {labels.mean():.3f}")
    
    # 计算简单指标
    binary_preds = (probs > 0.5).float()
    accuracy = (binary_preds == labels).float().mean()
    logger.info(f"\nBatch准确率: {accuracy:.3f}")
    
    return probs.mean().item()


def check_training_step(model, data_loader, device, learning_rate=3e-4):
    """检查训练步骤"""
    logger.info("\n" + "="*80)
    logger.info("检查训练步骤")
    logger.info("="*80)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练几个batch
    losses = []
    for i, batch in enumerate(data_loader):
        if i >= 5:
            break
        
        images = batch['image'].to(device)
        labels = batch['attr_labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs['attr_logits'], labels)
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        optimizer.step()
        
        losses.append(loss.item())
        logger.info(f"Batch {i+1}: Loss={loss.item():.4f}, Grad Norm={total_norm:.4f}")
    
    avg_loss = sum(losses) / len(losses)
    logger.info(f"\n平均损失: {avg_loss:.4f}")
    
    return avg_loss


def main():
    """主函数"""
    logger.info("\n" + "="*80)
    logger.info("实验调试 - 诊断F1分数过低问题")
    logger.info("="*80)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n使用设备: {device}")
    
    # 创建数据加载器
    logger.info("\n创建数据加载器...")
    try:
        train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
        logger.info(f"✓ 训练集大小: {len(train_loader.dataset)}")
        logger.info(f"✓ 验证集大小: {len(val_loader.dataset)}")
        logger.info(f"✓ Batch size: {EXPERIMENT_CONFIG['data']['batch_size']}")
    except Exception as e:
        logger.error(f"✗ 创建数据加载器失败: {str(e)}")
        return 1
    
    # 检查数据分布
    pos_ratio = check_data_distribution(train_loader)
    
    # 创建模型
    logger.info("\n创建模型...")
    model = CNNWithAdaptiveGAT(
        num_classes=26,
        cnn_type='resnet50',
        weights='IMAGENET1K_V1',
        lambda_threshold=0.5
    ).to(device)
    logger.info("✓ 模型创建成功")
    
    # 检查模型输出
    pred_ratio = check_model_output(model, val_loader, device)
    
    # 检查训练步骤
    avg_loss = check_training_step(model, train_loader, device)
    
    # 诊断结论
    logger.info("\n" + "="*80)
    logger.info("诊断结论")
    logger.info("="*80)
    
    issues = []
    
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        issues.append(f"⚠ 数据不平衡严重 (正样本比例: {pos_ratio:.3f})")
    
    if pred_ratio < 0.1 or pred_ratio > 0.9:
        issues.append(f"⚠ 模型预测偏向性强 (预测比例: {pred_ratio:.3f})")
    
    if avg_loss > 1.0:
        issues.append(f"⚠ 初始损失过高 (损失: {avg_loss:.3f})")
    
    if len(train_loader.dataset) < 100:
        issues.append(f"⚠ 训练集样本过少 (样本数: {len(train_loader.dataset)})")
    
    if issues:
        logger.info("\n发现以下问题:")
        for issue in issues:
            logger.info(f"  {issue}")
        
        logger.info("\n建议:")
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            logger.info("  1. 考虑使用加权损失函数处理类别不平衡")
            logger.info("  2. 使用Focal Loss或调整正负样本权重")
        
        if len(train_loader.dataset) < 100:
            logger.info("  3. 检查数据集路径是否正确")
            logger.info("  4. 确认数据预处理是否正确")
        
        if avg_loss > 1.0:
            logger.info("  5. 降低学习率")
            logger.info("  6. 检查模型初始化")
    else:
        logger.info("\n✓ 未发现明显问题，可能需要:")
        logger.info("  1. 增加训练轮数")
        logger.info("  2. 调整学习率")
        logger.info("  3. 使用学习率预热")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()

