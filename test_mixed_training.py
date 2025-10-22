#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合训练系统测试脚本

测试TextileNet数据集加载和混合数据集功能
"""

import os
import torch
import logging
from pathlib import Path
from torchvision import transforms

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入我们的模块
from training import TextileNetDataset, MixedDataset, get_transforms
from base_model import FullModel


def test_textile_dataset():
    """测试TextileNet数据集加载"""
    logger.info("测试TextileNet数据集加载...")
    
    textile_root = "/home/cv_model"
    transform = get_transforms(train=True)
    
    # 测试Fabric数据集
    try:
        fabric_dataset = TextileNetDataset(
            root_dir=textile_root,
            dataset_type='fabric',
            split='train',
            transform=transform
        )
        logger.info(f"✓ Fabric数据集加载成功: {len(fabric_dataset)} 样本")
        logger.info(f"  类别数: {fabric_dataset.get_num_classes()}")
        logger.info(f"  类别名: {fabric_dataset.get_class_names()[:5]}...")  # 显示前5个类别
        
        # 测试获取样本
        sample = fabric_dataset[0]
        logger.info(f"  样本键: {list(sample.keys())}")
        logger.info(f"  图像形状: {sample['image'].shape}")
        
    except Exception as e:
        logger.error(f"✗ Fabric数据集加载失败: {e}")
    
    # 测试Fiber数据集
    try:
        fiber_dataset = TextileNetDataset(
            root_dir=textile_root,
            dataset_type='fiber',
            split='train',
            transform=transform
        )
        logger.info(f"✓ Fiber数据集加载成功: {len(fiber_dataset)} 样本")
        logger.info(f"  类别数: {fiber_dataset.get_num_classes()}")
        logger.info(f"  类别名: {fiber_dataset.get_class_names()[:5]}...")  # 显示前5个类别
        
        # 测试获取样本
        sample = fiber_dataset[0]
        logger.info(f"  样本键: {list(sample.keys())}")
        logger.info(f"  图像形状: {sample['image'].shape}")
        
        return fabric_dataset, fiber_dataset
        
    except Exception as e:
        logger.error(f"✗ Fiber数据集加载失败: {e}")
        return None, None


def test_mixed_dataset(fabric_dataset, fiber_dataset):
    """测试混合数据集"""
    if fabric_dataset is None or fiber_dataset is None:
        logger.warning("跳过混合数据集测试（缺少基础数据集）")
        return None
    
    logger.info("测试混合数据集...")
    
    try:
        mixed_dataset = MixedDataset(
            deepfashion_dataset=None,  # 暂时不使用DeepFashion
            fabric_dataset=fabric_dataset,
            fiber_dataset=fiber_dataset,
            mixing_strategy='balanced',
            deepfashion_weight=0.0,
            fabric_weight=0.5,
            fiber_weight=0.5
        )
        
        logger.info(f"✓ 混合数据集创建成功: {len(mixed_dataset)} 样本")
        
        # 获取数据集信息
        info = mixed_dataset.get_dataset_info()
        logger.info("数据集信息:")
        for name, dataset_info in info['datasets'].items():
            logger.info(f"  {name}: {dataset_info}")
        
        # 测试获取样本
        sample = mixed_dataset[0]
        logger.info(f"样本键: {list(sample.keys())}")
        logger.info(f"来源数据集: {sample.get('source_dataset')}")
        logger.info(f"图像形状: {sample['image'].shape}")
        
        return mixed_dataset
        
    except Exception as e:
        logger.error(f"✗ 混合数据集创建失败: {e}")
        return None


def test_model(fabric_dataset, fiber_dataset):
    """测试增强模型"""
    logger.info("测试增强模型...")
    
    try:
        # 获取类别数量
        num_fabric_classes = fabric_dataset.get_num_classes() if fabric_dataset else 20
        num_fiber_classes = fiber_dataset.get_num_classes() if fiber_dataset else 32
        
        # 创建模型
        model = FullModel(
            num_classes=26,
            enable_segmentation=False,
            enable_textile_classification=True,
            num_fabric_classes=num_fabric_classes,
            num_fiber_classes=num_fiber_classes,
            gat_dims=[512, 256],  # 使用较小的维度进行测试
            gat_heads=4
        )
        
        logger.info(f"✓ 模型创建成功")
        logger.info(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            outputs = model(test_input)
        
        logger.info("模型输出:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")
        
        # 验证输出维度
        expected_shapes = {
            'attr_logits': (batch_size, 26),
            'fabric_logits': (batch_size, num_fabric_classes),
            'fiber_logits': (batch_size, num_fiber_classes),
            'textile_logits': (batch_size, max(num_fabric_classes, num_fiber_classes))
        }
        
        for key, expected_shape in expected_shapes.items():
            if key in outputs:
                actual_shape = outputs[key].shape
                if actual_shape == expected_shape:
                    logger.info(f"  ✓ {key} 形状正确: {actual_shape}")
                else:
                    logger.error(f"  ✗ {key} 形状错误: 期望 {expected_shape}, 实际 {actual_shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_paths():
    """测试数据路径"""
    logger.info("检查数据路径...")
    
    paths_to_check = [
        "/home/cv_model/fabric/train",
        "/home/cv_model/fiber/train",
        "/home/cv_model/DeepFashion"
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            logger.info(f"✓ 路径存在: {path}")
            if os.path.isdir(path):
                try:
                    contents = os.listdir(path)
                    logger.info(f"  包含 {len(contents)} 个项目")
                    if contents:
                        logger.info(f"  示例内容: {contents[:3]}...")
                except PermissionError:
                    logger.warning(f"  无法读取目录内容（权限不足）")
        else:
            logger.warning(f"✗ 路径不存在: {path}")


def main():
    """主测试函数"""
    logger.info("开始混合训练系统测试...")
    logger.info("=" * 50)
    
    # 1. 测试数据路径
    test_data_paths()
    logger.info("=" * 50)
    
    # 2. 测试TextileNet数据集
    fabric_dataset, fiber_dataset = test_textile_dataset()
    logger.info("=" * 50)
    
    # 3. 测试混合数据集
    mixed_dataset = test_mixed_dataset(fabric_dataset, fiber_dataset)
    logger.info("=" * 50)
    
    # 4. 测试模型
    model = test_model(fabric_dataset, fiber_dataset)
    logger.info("=" * 50)
    
    # 5. 总结
    logger.info("测试总结:")
    logger.info(f"  TextileNet数据集: {'✓' if fabric_dataset and fiber_dataset else '✗'}")
    logger.info(f"  混合数据集: {'✓' if mixed_dataset else '✗'}")
    logger.info(f"  增强模型: {'✓' if model else '✗'}")
    
    if fabric_dataset and fiber_dataset and mixed_dataset and model:
        logger.info("🎉 所有测试通过！系统准备就绪。")
        return True
    else:
        logger.warning("⚠️  部分测试失败，请检查配置。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
