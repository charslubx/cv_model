#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整融合的混合数据集训练脚本

基于现有的training.py，完整融合混合数据集功能进行训练
整合DeepFashion和TextileNet数据集的多任务学习

作者: AI Assistant
日期: 2025-10-23
"""

import os
import sys
import json
import copy
import random
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PIL import Image, ImageFile
from tqdm import tqdm
from sklearn.model_selection import KFold

# 导入基础模型和训练组件
from base_model import FullModel, FocalLoss, MultiTaskLoss
from training import (
    TextileNetDataset, MixedDataset, MixedDatasetTrainer,
    DeepFashionDataset, get_transforms, collate_fn
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('integrated_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置常量
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 设置随机种子
def set_seed(seed=42):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


class IntegratedTrainingConfig:
    """集成训练配置类"""
    
    def __init__(self):
        # 数据集路径配置
        self.project_root = "/home/cv_model"
        
        # DeepFashion配置
        self.deepfashion_root = os.path.join(self.project_root, "DeepFashion")
        self.deepfashion_category_root = os.path.join(
            self.deepfashion_root, "Category and Attribute Prediction Benchmark"
        )
        self.deepfashion_img_dir = os.path.join(self.deepfashion_category_root, "Img", "img")
        self.deepfashion_anno_dir = os.path.join(self.deepfashion_category_root, "Anno_fine")
        
        # TextileNet配置
        self.textile_root = self.project_root
        self.fabric_root = os.path.join(self.textile_root, "fabric")
        self.fiber_root = os.path.join(self.textile_root, "fiber")
        
        # 训练配置
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.epochs = 50
        self.num_workers = 4
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 模型配置
        self.num_deepfashion_attrs = 26
        self.enable_segmentation = False
        self.enable_textile_classification = True
        
        # 混合数据集配置
        self.mixing_strategy = 'balanced'  # 'balanced', 'weighted', 'sequential'
        self.deepfashion_weight = 0.5
        self.fabric_weight = 0.25
        self.fiber_weight = 0.25
        
        # 保存配置
        self.save_dir = "integrated_checkpoints"
        self.log_interval = 10
        
        # GAT配置
        self.gat_dims = [1024, 512]
        self.gat_heads = 8
        self.gat_dropout = 0.2
        
        logger.info("训练配置初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"批次大小: {self.batch_size}")
        logger.info(f"学习率: {self.learning_rate}")
        logger.info(f"训练轮数: {self.epochs}")


class DatasetManager:
    """数据集管理器"""
    
    def __init__(self, config: IntegratedTrainingConfig):
        self.config = config
        self.datasets = {}
        self.data_loaders = {}
        
    def setup_transforms(self):
        """设置数据增强"""
        self.train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info("数据增强设置完成")
    
    def load_deepfashion_datasets(self):
        """加载DeepFashion数据集"""
        try:
            if not os.path.exists(self.config.deepfashion_root):
                logger.warning(f"DeepFashion路径不存在: {self.config.deepfashion_root}")
                return None, None
            
            # 定义文件路径
            train_img_list = os.path.join(self.config.deepfashion_anno_dir, "train.txt")
            train_attr_file = os.path.join(self.config.deepfashion_anno_dir, "train_attr.txt")
            val_img_list = os.path.join(self.config.deepfashion_anno_dir, "val.txt")
            val_attr_file = os.path.join(self.config.deepfashion_anno_dir, "val_attr.txt")
            
            # 检查文件是否存在
            required_files = [train_img_list, train_attr_file, val_img_list, val_attr_file]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.warning(f"DeepFashion文件不存在: {file_path}")
                    return None, None
            
            # 创建数据集
            deepfashion_train = DeepFashionDataset(
                img_list_file=train_img_list,
                attr_file=train_attr_file,
                image_dir=self.config.deepfashion_img_dir,
                transform=self.train_transform
            )
            
            deepfashion_val = DeepFashionDataset(
                img_list_file=val_img_list,
                attr_file=val_attr_file,
                image_dir=self.config.deepfashion_img_dir,
                transform=self.val_transform
            )
            
            logger.info(f"DeepFashion数据集加载成功:")
            logger.info(f"  训练集: {len(deepfashion_train)} 样本")
            logger.info(f"  验证集: {len(deepfashion_val)} 样本")
            
            return deepfashion_train, deepfashion_val
            
        except Exception as e:
            logger.error(f"DeepFashion数据集加载失败: {e}")
            return None, None
    
    def load_textile_datasets(self):
        """加载TextileNet数据集"""
        fabric_train, fabric_val = None, None
        fiber_train, fiber_val = None, None
        
        # 加载Fabric数据集
        try:
            if os.path.exists(os.path.join(self.config.fabric_root, "train")):
                fabric_train = TextileNetDataset(
                    root_dir=self.config.textile_root,
                    dataset_type='fabric',
                    split='train',
                    transform=self.train_transform
                )
                
                # 检查是否有test分割
                fabric_test_dir = os.path.join(self.config.fabric_root, "test")
                if os.path.exists(fabric_test_dir) and os.listdir(fabric_test_dir):
                    fabric_val = TextileNetDataset(
                        root_dir=self.config.textile_root,
                        dataset_type='fabric',
                        split='test',
                        transform=self.val_transform
                    )
                else:
                    fabric_val = fabric_train  # 使用训练集作为验证集
                
                logger.info(f"Fabric数据集加载成功:")
                logger.info(f"  训练集: {len(fabric_train)} 样本, {fabric_train.get_num_classes()} 类别")
                logger.info(f"  验证集: {len(fabric_val)} 样本")
                logger.info(f"  类别: {fabric_train.get_class_names()}")
                
        except Exception as e:
            logger.warning(f"Fabric数据集加载失败: {e}")
        
        # 加载Fiber数据集
        try:
            if os.path.exists(os.path.join(self.config.fiber_root, "train")):
                fiber_train = TextileNetDataset(
                    root_dir=self.config.textile_root,
                    dataset_type='fiber',
                    split='train',
                    transform=self.train_transform
                )
                
                # 检查是否有test分割
                fiber_test_dir = os.path.join(self.config.fiber_root, "test")
                if os.path.exists(fiber_test_dir) and os.listdir(fiber_test_dir):
                    fiber_val = TextileNetDataset(
                        root_dir=self.config.textile_root,
                        dataset_type='fiber',
                        split='test',
                        transform=self.val_transform
                    )
                else:
                    fiber_val = fiber_train  # 使用训练集作为验证集
                
                logger.info(f"Fiber数据集加载成功:")
                logger.info(f"  训练集: {len(fiber_train)} 样本, {fiber_train.get_num_classes()} 类别")
                logger.info(f"  验证集: {len(fiber_val)} 样本")
                logger.info(f"  类别: {fiber_train.get_class_names()}")
                
        except Exception as e:
            logger.warning(f"Fiber数据集加载失败: {e}")
        
        return fabric_train, fabric_val, fiber_train, fiber_val
    
    def create_mixed_datasets(self, deepfashion_train, deepfashion_val, 
                            fabric_train, fabric_val, fiber_train, fiber_val):
        """创建混合数据集"""
        try:
            # 创建混合训练数据集
            mixed_train = MixedDataset(
                deepfashion_dataset=deepfashion_train,
                fabric_dataset=fabric_train,
                fiber_dataset=fiber_train,
                mixing_strategy=self.config.mixing_strategy,
                deepfashion_weight=self.config.deepfashion_weight,
                fabric_weight=self.config.fabric_weight,
                fiber_weight=self.config.fiber_weight
            )
            
            # 创建混合验证数据集
            mixed_val = MixedDataset(
                deepfashion_dataset=deepfashion_val,
                fabric_dataset=fabric_val,
                fiber_dataset=fiber_val,
                mixing_strategy=self.config.mixing_strategy,
                deepfashion_weight=self.config.deepfashion_weight,
                fabric_weight=self.config.fabric_weight,
                fiber_weight=self.config.fiber_weight
            )
            
            # 打印数据集信息
            logger.info("混合数据集创建成功:")
            train_info = mixed_train.get_dataset_info()
            logger.info(f"  训练集总长度: {train_info['total_length']}")
            logger.info(f"  混合策略: {train_info['mixing_strategy']}")
            
            for dataset_name, info in train_info['datasets'].items():
                logger.info(f"  {dataset_name}: 长度={info['length']}, 权重={info['weight']}")
            
            return mixed_train, mixed_val
            
        except Exception as e:
            logger.error(f"混合数据集创建失败: {e}")
            raise
    
    def create_data_loaders(self, train_dataset, val_dataset):
        """创建数据加载器"""
        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn
            )
            
            logger.info("数据加载器创建成功:")
            logger.info(f"  训练批次数: {len(train_loader)}")
            logger.info(f"  验证批次数: {len(val_loader)}")
            
            return train_loader, val_loader
            
        except Exception as e:
            logger.error(f"数据加载器创建失败: {e}")
            raise
    
    def setup_all_datasets(self):
        """设置所有数据集"""
        logger.info("开始设置数据集...")
        
        # 1. 设置数据增强
        self.setup_transforms()
        
        # 2. 加载DeepFashion数据集
        deepfashion_train, deepfashion_val = self.load_deepfashion_datasets()
        
        # 3. 加载TextileNet数据集
        fabric_train, fabric_val, fiber_train, fiber_val = self.load_textile_datasets()
        
        # 4. 检查是否至少有一个数据集可用
        available_datasets = []
        if deepfashion_train is not None:
            available_datasets.append("DeepFashion")
        if fabric_train is not None:
            available_datasets.append("Fabric")
        if fiber_train is not None:
            available_datasets.append("Fiber")
        
        if not available_datasets:
            raise ValueError("没有可用的数据集！请检查数据路径。")
        
        logger.info(f"可用数据集: {available_datasets}")
        
        # 5. 创建混合数据集
        mixed_train, mixed_val = self.create_mixed_datasets(
            deepfashion_train, deepfashion_val,
            fabric_train, fabric_val,
            fiber_train, fiber_val
        )
        
        # 6. 创建数据加载器
        train_loader, val_loader = self.create_data_loaders(mixed_train, mixed_val)
        
        # 7. 保存数据集信息
        self.datasets = {
            'deepfashion_train': deepfashion_train,
            'deepfashion_val': deepfashion_val,
            'fabric_train': fabric_train,
            'fabric_val': fabric_val,
            'fiber_train': fiber_train,
            'fiber_val': fiber_val,
            'mixed_train': mixed_train,
            'mixed_val': mixed_val
        }
        
        self.data_loaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        # 8. 获取类别数量信息
        self.num_fabric_classes = fabric_train.get_num_classes() if fabric_train else 20
        self.num_fiber_classes = fiber_train.get_num_classes() if fiber_train else 32
        
        logger.info("数据集设置完成!")
        logger.info(f"Fabric类别数: {self.num_fabric_classes}")
        logger.info(f"Fiber类别数: {self.num_fiber_classes}")
        
        return train_loader, val_loader


class ModelManager:
    """模型管理器"""
    
    def __init__(self, config: IntegratedTrainingConfig, dataset_manager: DatasetManager):
        self.config = config
        self.dataset_manager = dataset_manager
        self.model = None
        self.trainer = None
    
    def create_model(self):
        """创建模型"""
        try:
            logger.info("创建增强模型...")
            
            self.model = FullModel(
                num_classes=self.config.num_deepfashion_attrs,
                cnn_type='resnet50',
                weights='IMAGENET1K_V1',
                enable_segmentation=self.config.enable_segmentation,
                enable_textile_classification=self.config.enable_textile_classification,
                num_fabric_classes=self.dataset_manager.num_fabric_classes,
                num_fiber_classes=self.dataset_manager.num_fiber_classes,
                gat_dims=self.config.gat_dims,
                gat_heads=self.config.gat_heads,
                gat_dropout=self.config.gat_dropout
            )
            
            # 移动到设备
            self.model = self.model.to(self.config.device)
            
            # 计算参数数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info("模型创建成功:")
            logger.info(f"  总参数数: {total_params:,}")
            logger.info(f"  可训练参数数: {trainable_params:,}")
            logger.info(f"  设备: {self.config.device}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"模型创建失败: {e}")
            raise
    
    def create_trainer(self):
        """创建训练器"""
        try:
            if self.model is None:
                raise ValueError("模型未创建，请先调用create_model()")
            
            logger.info("创建混合数据集训练器...")
            
            self.trainer = MixedDatasetTrainer(
                model=self.model,
                train_loader=self.dataset_manager.data_loaders['train'],
                val_loader=self.dataset_manager.data_loaders['val'],
                device=self.config.device,
                learning_rate=self.config.learning_rate,
                enable_segmentation=self.config.enable_segmentation,
                enable_textile_classification=self.config.enable_textile_classification
            )
            
            logger.info("训练器创建成功")
            return self.trainer
            
        except Exception as e:
            logger.error(f"训练器创建失败: {e}")
            raise


class TrainingManager:
    """训练管理器"""
    
    def __init__(self, config: IntegratedTrainingConfig):
        self.config = config
        self.dataset_manager = DatasetManager(config)
        self.model_manager = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def setup(self):
        """设置训练环境"""
        logger.info("设置训练环境...")
        
        # 1. 创建保存目录
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # 2. 设置数据集
        train_loader, val_loader = self.dataset_manager.setup_all_datasets()
        
        # 3. 创建模型管理器
        self.model_manager = ModelManager(self.config, self.dataset_manager)
        
        # 4. 创建模型和训练器
        model = self.model_manager.create_model()
        trainer = self.model_manager.create_trainer()
        
        logger.info("训练环境设置完成")
        return trainer
    
    def train(self):
        """执行完整训练流程"""
        try:
            logger.info("开始完整训练流程...")
            
            # 1. 设置训练环境
            trainer = self.setup()
            
            # 2. 保存配置
            self.save_config()
            
            # 3. 开始训练
            logger.info(f"开始训练 {self.config.epochs} 轮...")
            trainer.train(epochs=self.config.epochs, save_dir=self.config.save_dir)
            
            # 4. 保存训练历史
            self.save_training_history()
            
            logger.info("训练完成!")
            
        except Exception as e:
            logger.error(f"训练失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def save_config(self):
        """保存训练配置"""
        try:
            config_dict = {
                'project_root': self.config.project_root,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs,
                'device': self.config.device,
                'mixing_strategy': self.config.mixing_strategy,
                'deepfashion_weight': self.config.deepfashion_weight,
                'fabric_weight': self.config.fabric_weight,
                'fiber_weight': self.config.fiber_weight,
                'gat_dims': self.config.gat_dims,
                'gat_heads': self.config.gat_heads,
                'num_fabric_classes': self.dataset_manager.num_fabric_classes,
                'num_fiber_classes': self.dataset_manager.num_fiber_classes
            }
            
            config_path = os.path.join(self.config.save_dir, "training_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练配置已保存到: {config_path}")
            
        except Exception as e:
            logger.warning(f"保存配置失败: {e}")
    
    def save_training_history(self):
        """保存训练历史"""
        try:
            history_path = os.path.join(self.config.save_dir, "training_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
            
            logger.info(f"训练历史已保存到: {history_path}")
            
        except Exception as e:
            logger.warning(f"保存训练历史失败: {e}")


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("开始完整融合的混合数据集训练")
    logger.info("=" * 80)
    
    try:
        # 1. 创建配置
        config = IntegratedTrainingConfig()
        
        # 2. 创建训练管理器
        training_manager = TrainingManager(config)
        
        # 3. 执行训练
        training_manager.train()
        
        logger.info("=" * 80)
        logger.info("训练成功完成!")
        logger.info(f"模型保存在: {config.save_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error("=" * 80)
        logger.error("训练失败!")
        logger.error(f"错误: {e}")
        logger.error("=" * 80)
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
