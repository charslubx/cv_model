#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合数据集训练脚本
结合DeepFashion和TextileNet数据集进行多任务学习

作者: AI Assistant
日期: 2025-10-22
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training import (
    DeepFashionDataset, TextileNetDataset, MixedDataset, 
    MixedDatasetTrainer, get_transforms, collate_fn
)
from base_model import FullModel
from torch.utils.data import DataLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_datasets(config):
    """设置数据集"""
    logger.info("正在设置数据集...")
    
    # 数据增强
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    datasets = {}
    
    # 1. DeepFashion数据集（如果路径存在）
    if os.path.exists(config['deepfashion_root']):
        logger.info("加载DeepFashion数据集...")
        try:
            deepfashion_train = DeepFashionDataset(
                img_list_file=config['deepfashion_train_list'],
                attr_file=config['deepfashion_train_attr'],
                image_dir=config['deepfashion_img_dir'],
                transform=train_transform
            )
            
            deepfashion_val = DeepFashionDataset(
                img_list_file=config['deepfashion_val_list'],
                attr_file=config['deepfashion_val_attr'],
                image_dir=config['deepfashion_img_dir'],
                transform=val_transform
            )
            
            datasets['deepfashion_train'] = deepfashion_train
            datasets['deepfashion_val'] = deepfashion_val
            logger.info(f"DeepFashion训练集: {len(deepfashion_train)} 样本")
            logger.info(f"DeepFashion验证集: {len(deepfashion_val)} 样本")
            
        except Exception as e:
            logger.warning(f"DeepFashion数据集加载失败: {e}")
            datasets['deepfashion_train'] = None
            datasets['deepfashion_val'] = None
    else:
        logger.warning(f"DeepFashion路径不存在: {config['deepfashion_root']}")
        datasets['deepfashion_train'] = None
        datasets['deepfashion_val'] = None
    
    # 2. Fabric数据集
    if os.path.exists(config['fabric_root']):
        logger.info("加载Fabric数据集...")
        try:
            fabric_train = TextileNetDataset(
                root_dir=config['textile_root'],
                dataset_type='fabric',
                split='train',
                transform=train_transform
            )
            
            # 检查是否有test分割，否则使用train的一部分作为验证
            fabric_test_dir = os.path.join(config['fabric_root'], 'test')
            if os.path.exists(fabric_test_dir) and os.listdir(fabric_test_dir):
                fabric_val = TextileNetDataset(
                    root_dir=config['textile_root'],
                    dataset_type='fabric',
                    split='test',
                    transform=val_transform
                )
            else:
                # 使用训练集的一部分作为验证集
                fabric_val = fabric_train
                logger.info("Fabric数据集没有test分割，使用train作为验证集")
            
            datasets['fabric_train'] = fabric_train
            datasets['fabric_val'] = fabric_val
            logger.info(f"Fabric训练集: {len(fabric_train)} 样本")
            logger.info(f"Fabric验证集: {len(fabric_val)} 样本")
            
        except Exception as e:
            logger.warning(f"Fabric数据集加载失败: {e}")
            datasets['fabric_train'] = None
            datasets['fabric_val'] = None
    else:
        logger.warning(f"Fabric路径不存在: {config['fabric_root']}")
        datasets['fabric_train'] = None
        datasets['fabric_val'] = None
    
    # 3. Fiber数据集
    if os.path.exists(config['fiber_root']):
        logger.info("加载Fiber数据集...")
        try:
            fiber_train = TextileNetDataset(
                root_dir=config['textile_root'],
                dataset_type='fiber',
                split='train',
                transform=train_transform
            )
            
            # 检查是否有test分割
            fiber_test_dir = os.path.join(config['fiber_root'], 'test')
            if os.path.exists(fiber_test_dir) and os.listdir(fiber_test_dir):
                fiber_val = TextileNetDataset(
                    root_dir=config['textile_root'],
                    dataset_type='fiber',
                    split='test',
                    transform=val_transform
                )
            else:
                # 使用训练集的一部分作为验证集
                fiber_val = fiber_train
                logger.info("Fiber数据集没有test分割，使用train作为验证集")
            
            datasets['fiber_train'] = fiber_train
            datasets['fiber_val'] = fiber_val
            logger.info(f"Fiber训练集: {len(fiber_train)} 样本")
            logger.info(f"Fiber验证集: {len(fiber_val)} 样本")
            
        except Exception as e:
            logger.warning(f"Fiber数据集加载失败: {e}")
            datasets['fiber_train'] = None
            datasets['fiber_val'] = None
    else:
        logger.warning(f"Fiber路径不存在: {config['fiber_root']}")
        datasets['fiber_train'] = None
        datasets['fiber_val'] = None
    
    return datasets


def create_mixed_datasets(datasets, config):
    """创建混合数据集"""
    logger.info("创建混合数据集...")
    
    # 创建混合训练数据集
    mixed_train = MixedDataset(
        deepfashion_dataset=datasets.get('deepfashion_train'),
        fabric_dataset=datasets.get('fabric_train'),
        fiber_dataset=datasets.get('fiber_train'),
        mixing_strategy=config['mixing_strategy'],
        deepfashion_weight=config['deepfashion_weight'],
        fabric_weight=config['fabric_weight'],
        fiber_weight=config['fiber_weight']
    )
    
    # 创建混合验证数据集
    mixed_val = MixedDataset(
        deepfashion_dataset=datasets.get('deepfashion_val'),
        fabric_dataset=datasets.get('fabric_val'),
        fiber_dataset=datasets.get('fiber_val'),
        mixing_strategy=config['mixing_strategy'],
        deepfashion_weight=config['deepfashion_weight'],
        fabric_weight=config['fabric_weight'],
        fiber_weight=config['fiber_weight']
    )
    
    # 打印混合数据集信息
    logger.info("混合训练数据集信息:")
    train_info = mixed_train.get_dataset_info()
    for dataset_name, info in train_info['datasets'].items():
        logger.info(f"  - {dataset_name}: {info}")
    
    logger.info("混合验证数据集信息:")
    val_info = mixed_val.get_dataset_info()
    for dataset_name, info in val_info['datasets'].items():
        logger.info(f"  - {dataset_name}: {info}")
    
    return mixed_train, mixed_val


def main():
    """主函数"""
    logger.info("开始混合数据集训练...")
    
    # 配置参数
    config = {
        # DeepFashion路径
        'deepfashion_root': '/home/cv_model/DeepFashion',
        'deepfashion_img_dir': '/home/cv_model/DeepFashion/Category and Attribute Prediction Benchmark/Img/img',
        'deepfashion_train_list': '/home/cv_model/DeepFashion/Category and Attribute Prediction Benchmark/Anno_fine/train.txt',
        'deepfashion_train_attr': '/home/cv_model/DeepFashion/Category and Attribute Prediction Benchmark/Anno_fine/train_attr.txt',
        'deepfashion_val_list': '/home/cv_model/DeepFashion/Category and Attribute Prediction Benchmark/Anno_fine/val.txt',
        'deepfashion_val_attr': '/home/cv_model/DeepFashion/Category and Attribute Prediction Benchmark/Anno_fine/val_attr.txt',
        
        # TextileNet路径
        'textile_root': '/home/cv_model',
        'fabric_root': '/home/cv_model/fabric',
        'fiber_root': '/home/cv_model/fiber',
        
        # 混合策略
        'mixing_strategy': 'balanced',
        'deepfashion_weight': 0.5,
        'fabric_weight': 0.25,
        'fiber_weight': 0.25,
        
        # 训练参数
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 50,
        'save_dir': 'mixed_checkpoints'
    }
    
    try:
        # 1. 设置数据集
        datasets = setup_datasets(config)
        
        # 检查是否至少有一个数据集可用
        available_datasets = [name for name, dataset in datasets.items() 
                            if dataset is not None and 'train' in name]
        
        if not available_datasets:
            logger.error("没有可用的训练数据集！请检查数据路径。")
            return
        
        logger.info(f"可用的训练数据集: {available_datasets}")
        
        # 2. 创建混合数据集
        mixed_train, mixed_val = create_mixed_datasets(datasets, config)
        
        # 3. 创建数据加载器
        train_loader = DataLoader(
            mixed_train,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            mixed_val,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        # 4. 获取类别数量
        num_fabric_classes = datasets['fabric_train'].get_num_classes() if datasets['fabric_train'] else 20
        num_fiber_classes = datasets['fiber_train'].get_num_classes() if datasets['fiber_train'] else 32
        
        logger.info(f"Fabric类别数: {num_fabric_classes}")
        logger.info(f"Fiber类别数: {num_fiber_classes}")
        
        # 5. 创建模型
        model = FullModel(
            num_classes=26,  # DeepFashion属性数量
            enable_segmentation=False,  # 暂时禁用分割
            enable_textile_classification=True,  # 启用纹理分类
            num_fabric_classes=num_fabric_classes,
            num_fiber_classes=num_fiber_classes,
            gat_dims=[1024, 512],
            gat_heads=8,
            cnn_type='resnet50',
            weights='IMAGENET1K_V1'
        )
        
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 6. 创建训练器
        trainer = MixedDatasetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            enable_segmentation=False,
            enable_textile_classification=True,
            learning_rate=config['learning_rate']
        )
        
        # 7. 开始训练
        logger.info("开始混合数据集训练...")
        trainer.train(
            epochs=config['epochs'], 
            save_dir=config['save_dir']
        )
        
        logger.info("混合数据集训练完成！")
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
