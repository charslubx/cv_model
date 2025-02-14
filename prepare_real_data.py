import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import logging
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_dataset(source_dir, target_dir="data/images", num_classes=20, min_samples=100):
    """准备真实图片数据集
    
    Args:
        source_dir: 源图片目录(包含多个类别子目录)
        target_dir: 目标图片目录
        num_classes: 需要的类别数
        min_samples: 每个类别最少样本数
    """
    try:
        # 创建目录
        os.makedirs(target_dir, exist_ok=True)
        
        # 加载源数据集
        dataset = ImageFolder(source_dir)
        classes = dataset.classes
        
        # 确保有足够的类别
        if len(classes) < num_classes:
            raise ValueError(f"源数据集类别数({len(classes)})小于所需类别数({num_classes})")
        
        # 选择类别
        selected_classes = classes[:num_classes]
        logger.info(f"选择的类别: {selected_classes}")
        
        # 准备数据
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(tqdm(selected_classes)):
            class_dir = os.path.join(source_dir, class_name)
            images = os.listdir(class_dir)
            
            # 确保每个类别有足够的样本
            if len(images) < min_samples:
                raise ValueError(f"类别 {class_name} 样本数({len(images)})小于最小要求({min_samples})")
            
            # 随机选择样本
            selected_images = np.random.choice(images, min_samples, replace=False)
            
            for img_name in selected_images:
                src_path = os.path.join(class_dir, img_name)
                dst_path = os.path.join(target_dir, f"{class_name}_{img_name}")
                
                # 复制并转换图片
                try:
                    img = Image.open(src_path)
                    img = img.convert('RGB')
                    img.save(dst_path, 'JPEG')
                    
                    image_paths.append(os.path.basename(dst_path))
                    
                    # 创建one-hot标签
                    label = np.zeros(num_classes)
                    label[class_idx] = 1
                    labels.append(label)
                    
                except Exception as e:
                    logger.warning(f"处理图片 {src_path} 失败: {str(e)}")
                    continue
        
        # 创建标签文件
        df = pd.DataFrame(labels, columns=[f"class_{i}" for i in range(num_classes)])
        df['image_path'] = image_paths
        
        # 分割训练集和测试集
        indices = np.random.permutation(len(df))
        train_size = int(0.8 * len(df))
        
        train_df = df.iloc[indices[:train_size]]
        test_df = df.iloc[indices[train_size:]]
        
        train_df.to_csv("data/train_labels.csv", index=False)
        test_df.to_csv("data/test_labels.csv", index=False)
        
        logger.info(f"数据集准备完成:")
        logger.info(f"- 训练集大小: {len(train_df)}")
        logger.info(f"- 测试集大小: {len(test_df)}")
        logger.info(f"- 每个类别样本数: {min_samples}")
        
    except Exception as e:
        logger.error(f"数据准备失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 使用ImageNet或其他图片数据集目录
    source_dir = "path/to/your/image/dataset"  # 请替换为您的图片数据集路径
    prepare_dataset(source_dir) 