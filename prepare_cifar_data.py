import os
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL import Image
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_cifar10_dataset(target_dir="data/images", num_classes=10, samples_per_class=500):
    """准备CIFAR-10数据集
    
    Args:
        target_dir: 目标图片目录
        num_classes: 使用的类别数(最大10)
        samples_per_class: 每个类别的样本数
    """
    try:
        # 创建目录
        os.makedirs(target_dir, exist_ok=True)
        
        # 下载CIFAR-10
        logger.info("下载CIFAR-10数据集...")
        dataset = torchvision.datasets.CIFAR10(
            root='./data', 
            train=True, 
            download=True
        )
        
        # 准备数据
        image_paths = []
        labels = []
        
        logger.info("处理图片...")
        for class_idx in tqdm(range(num_classes)):
            # 获取当前类别的所有索引
            indices = np.where(np.array(dataset.targets) == class_idx)[0]
            
            # 随机选择样本
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
            
            for idx in selected_indices:
                img, _ = dataset[idx]
                img_name = f"class_{class_idx}_img_{idx}.jpg"
                img_path = os.path.join(target_dir, img_name)
                
                # 保存图片
                img.save(img_path)
                
                # 记录路径和标签
                image_paths.append(img_name)
                label = np.zeros(num_classes)
                label[class_idx] = 1
                labels.append(label)
        
        # 创建标签DataFrame
        df = pd.DataFrame(labels, columns=[f"class_{i}" for i in range(num_classes)])
        df['image_path'] = image_paths
        
        # 分割训练集和测试集
        indices = np.random.permutation(len(df))
        train_size = int(0.8 * len(df))
        
        train_df = df.iloc[indices[:train_size]]
        test_df = df.iloc[indices[train_size:]]
        
        # 保存标签文件
        train_df.to_csv("data/train_labels.csv", index=False)
        test_df.to_csv("data/test_labels.csv", index=False)
        
        logger.info(f"数据集准备完成:")
        logger.info(f"- 训练集大小: {len(train_df)}")
        logger.info(f"- 测试集大小: {len(test_df)}")
        logger.info(f"- 每个类别样本数: {samples_per_class}")
        
    except Exception as e:
        logger.error(f"数据准备失败: {str(e)}")
        raise