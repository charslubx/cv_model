"""
DeepFashion属性识别单独训练脚本
用于训练FullAdaGAT模型进行服装属性识别
"""

import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse

from ablation_models import FullAdaGAT
from training import DeepFashionDataset, DeepFashionTrainer

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='DeepFashion属性识别训练')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--save_dir', type=str, default='deepfashion_checkpoints', help='模型保存目录')
    parser.add_argument('--lambda_threshold', type=float, default=0.5, help='FullAdaGAT的lambda阈值')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("DeepFashion属性识别训练")
    logger.info("=" * 80)
    
    # 数据集路径
    DEEPFASHION_ROOT = "/home/cv_model/DeepFashion"
    CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
    ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")
    IMG_DIR = os.path.join(CATEGORY_ROOT, "Img")
    
    # 检查DeepFashion数据集
    if not os.path.exists(DEEPFASHION_ROOT):
        logger.error(f"DeepFashion数据集路径不存在: {DEEPFASHION_ROOT}")
        exit(1)
    
    # 检查必要文件
    required_files = [
        os.path.join(ANNO_DIR, "train.txt"),
        os.path.join(ANNO_DIR, "train_attr.txt"),
        os.path.join(ANNO_DIR, "val.txt"),
        os.path.join(ANNO_DIR, "val_attr.txt")
    ]
    
    if not all(os.path.exists(f) for f in required_files):
        logger.error("DeepFashion数据集文件不完整！")
        logger.error("缺失的文件:")
        for f in required_files:
            if not os.path.exists(f):
                logger.error(f"  - {f}")
        exit(1)
    
    logger.info("✓ DeepFashion数据集检查通过")
    
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载DeepFashion数据集
    logger.info("加载DeepFashion数据集...")
    TRAIN_IMG_LIST = os.path.join(ANNO_DIR, "train.txt")
    TRAIN_ATTR_FILE = os.path.join(ANNO_DIR, "train_attr.txt")
    VAL_IMG_LIST = os.path.join(ANNO_DIR, "val.txt")
    VAL_ATTR_FILE = os.path.join(ANNO_DIR, "val_attr.txt")
    
    train_dataset = DeepFashionDataset(
        img_list_file=TRAIN_IMG_LIST,
        attr_file=TRAIN_ATTR_FILE,
        image_dir=IMG_DIR,
        transform=train_transform
    )
    
    val_dataset = DeepFashionDataset(
        img_list_file=VAL_IMG_LIST,
        attr_file=VAL_ATTR_FILE,
        image_dir=IMG_DIR,
        transform=val_transform
    )
    
    logger.info(f"DeepFashion加载成功: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"数据加载器创建成功: 训练批次{len(train_loader)}, 验证批次{len(val_loader)}")
    
    # 创建FullAdaGAT模型
    logger.info("创建FullAdaGAT模型...")
    model = FullAdaGAT(
        num_classes=26,  # DeepFashion属性数量
        lambda_threshold=args.lambda_threshold
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建成功，参数数量: {total_params:,}")
    logger.info(f"Lambda阈值: {args.lambda_threshold}")
    
    # 创建训练器
    logger.info("创建训练器...")
    trainer = DeepFashionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr
    )
    
    # 开始训练
    logger.info("=" * 80)
    logger.info(f"开始训练...")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"保存目录: {args.save_dir}")
    logger.info("=" * 80)
    
    try:
        trainer.train(epochs=args.epochs, save_dir=args.save_dir)
        
        logger.info("=" * 80)
        logger.info("🎉 DeepFashion训练成功完成!")
        logger.info(f"模型已保存到: {args.save_dir}/best_model.pth")
        logger.info(f"最佳F1分数: {trainer.best_f1:.4f}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("❌ 训练过程中发生错误!")
        logger.error(f"错误信息: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()

