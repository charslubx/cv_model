import os
import pandas as pd
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import json
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_annotation_file(file_path):
    """读取标注文件"""
    annotations = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 1:
                    img_name = parts[0]
                    # 将属性标签转换为列表
                    attrs = parts[1:]
                    annotations[img_name] = attrs
    except Exception as e:
        logger.error(f"读取标注文件 {file_path} 失败: {str(e)}")
    return annotations

def merge_annotations(texture_ann_dir, shape_ann_dir, split):
    """合并纹理和形状标注"""
    merged_anns = {}
    
    # 读取纹理标注
    texture_types = ['lower_fused.txt', 'outer_fused.txt', 'upper_fused.txt']
    for txt_file in texture_types:
        file_path = os.path.join(texture_ann_dir, split, txt_file)
        if os.path.exists(file_path):
            texture_anns = read_annotation_file(file_path)
            for img_name, attrs in texture_anns.items():
                if img_name not in merged_anns:
                    merged_anns[img_name] = set()
                merged_anns[img_name].update(attrs)
    
    # 读取形状标注
    shape_file = os.path.join(shape_ann_dir, f'{split}_ann_file.txt')
    if os.path.exists(shape_file):
        shape_anns = read_annotation_file(shape_file)
        for img_name, attrs in shape_anns.items():
            if img_name not in merged_anns:
                merged_anns[img_name] = set()
            merged_anns[img_name].update(attrs)
    
    return merged_anns

def prepare_deepfashion_dataset(
    dataset_root,
    target_dir="data/images",
    min_samples_per_attr=100
):
    """准备DeepFashion数据集"""
    try:
        # 创建目录
        os.makedirs(target_dir, exist_ok=True)
        
        # 定义路径
        train_img_dir = os.path.join(dataset_root, "train_images")
        test_img_dir = os.path.join(dataset_root, "test_images")
        texture_ann_dir = os.path.join(dataset_root, "texture_ann")
        shape_ann_dir = os.path.join(dataset_root, "shape_ann")
        
        # 收集所有属性
        logger.info("收集属性集合...")
        all_attrs = set()
        for split in ['train', 'val']:
            anns = merge_annotations(texture_ann_dir, shape_ann_dir, split)
            for attrs in anns.values():
                all_attrs.update(attrs)
        
        # 将属性转换为有序列表
        attr_list = sorted(list(all_attrs))
        attr_to_idx = {attr: idx for idx, attr in enumerate(attr_list)}
        logger.info(f"总共发现 {len(attr_list)} 个属性")
        
        # 修改处理逻辑，分别处理训练集、验证集和测试集
        logger.info("处理训练、验证和测试集图片与标签...")
        splits_config = {
            'train': {'img_dir': train_img_dir, 'prefix': 'df_train_'},
            'val': {'img_dir': os.path.join(dataset_root, "densepose"), 'prefix': 'df_val_'},
            'test': {'img_dir': test_img_dir, 'prefix': 'df_test_'}
        }
        
        image_paths = []
        labels = []
        
        for split, config in splits_config.items():
            anns = merge_annotations(texture_ann_dir, shape_ann_dir, split)
            img_dir = config['img_dir']
            prefix = config['prefix']
            
            for img_name, attrs in tqdm(anns.items()):
                # 对于验证集，直接从densepose文件夹中查找图片
                src_path = os.path.join(img_dir, img_name)
                if not os.path.exists(src_path):
                    logger.warning(f"找不到图片: {src_path}")
                    continue
                
                # 创建标签向量
                label = np.zeros(len(attr_list))
                for attr in attrs:
                    if attr in attr_to_idx:
                        label[attr_to_idx[attr]] = 1
                
                # 使用新的前缀命名
                dst_name = f"{prefix}{os.path.basename(img_name)}"
                dst_path = os.path.join(target_dir, dst_name)
                
                try:
                    # 复制并转换图片
                    img = Image.open(src_path)
                    img = img.convert('RGB')
                    img.save(dst_path, 'JPEG')
                    
                    image_paths.append(dst_name)
                    labels.append(label)
                except Exception as e:
                    logger.warning(f"处理图片 {src_path} 失败: {str(e)}")
                    continue
        
        # 创建DataFrame
        df = pd.DataFrame(labels, columns=[f"attr_{i}" for i in range(len(attr_list))])
        df['image_path'] = image_paths
        
        # 过滤掉样本数太少的属性
        valid_attrs = []
        for col in df.columns:
            if col != 'image_path':
                pos_samples = df[col].sum()
                if pos_samples >= min_samples_per_attr:
                    valid_attrs.append(col)
                    attr_idx = int(col.split('_')[1])
                    logger.info(f"属性 {attr_list[attr_idx]} 的样本数: {pos_samples}")
        
        # 只保留有效属性
        df = df[['image_path'] + valid_attrs]
        
        # 分割训练集、验证集和测试集
        train_df = df[df['image_path'].str.contains('df_train_')]
        val_df = df[df['image_path'].str.contains('df_val_')]
        test_df = df[df['image_path'].str.contains('df_test_')]
        
        # 保存标签文件
        os.makedirs("data", exist_ok=True)
        train_df.to_csv("data/train_labels.csv", index=False)
        val_df.to_csv("data/val_labels.csv", index=False)
        test_df.to_csv("data/test_labels.csv", index=False)
        
        # 保存属性映射
        attr_map = {
            f"attr_{i}": name 
            for i, name in enumerate(attr_list) 
            if f"attr_{i}" in valid_attrs
        }
        pd.DataFrame.from_dict(attr_map, orient='index', columns=['name']).to_csv(
            "data/attribute_map.csv"
        )
        
        logger.info(f"数据集准备完成:")
        logger.info(f"- 训练集大小: {len(train_df)}")
        logger.info(f"- 验证集大小: {len(val_df)}")
        logger.info(f"- 测试集大小: {len(test_df)}")
        logger.info(f"- 有效属性数: {len(valid_attrs)}")
        
        return len(valid_attrs)
        
    except Exception as e:
        logger.error(f"数据准备失败: {str(e)}")
        raise

def load_attribute_names():
    """加载属性名称映射"""
    try:
        attr_map = pd.read_csv("data/attribute_map.csv")
        return attr_map['name'].tolist()
    except Exception as e:
        logger.error(f"加载属性映射失败: {str(e)}")
        return None 