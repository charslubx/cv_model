import os
import pandas as pd
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义数据集路径
DEEPFASHION_ROOT = "deepfashion"

# Category and Attribute Prediction Benchmark
CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
CATEGORY_IMG_DIR = os.path.join(CATEGORY_ROOT, "Img/img")
CATEGORY_ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")
CATEGORY_EVAL_DIR = os.path.join(CATEGORY_ROOT, "Eval")

# In-shop Clothes Retrieval Benchmark
INSHOP_ROOT = os.path.join(DEEPFASHION_ROOT, "In-shop Clothes Retrieval Benchmark")
INSHOP_IMG_DIR = os.path.join(INSHOP_ROOT, "Img/img")
INSHOP_ANNO_DIR = os.path.join(INSHOP_ROOT, "Anno")
INSHOP_EVAL_DIR = os.path.join(INSHOP_ROOT, "Eval")

# Consumer-to-shop Clothes Retrieval Benchmark
CONSUMER2SHOP_ROOT = os.path.join(DEEPFASHION_ROOT, "Consumer-to-shop Clothes Retrieval Benchmark")
CONSUMER2SHOP_IMG_DIR = os.path.join(CONSUMER2SHOP_ROOT, "Img/img")
CONSUMER2SHOP_ANNO_DIR = os.path.join(CONSUMER2SHOP_ROOT, "Anno")
CONSUMER2SHOP_EVAL_DIR = os.path.join(CONSUMER2SHOP_ROOT, "Eval")

def read_attr_cloth_file(file_path):
    """读取属性定义文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是属性数量
            num_attrs = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是属性定义
            attrs = []
            for line in lines[2:]:
                parts = line.strip().split()
                attr_name = parts[0]
                attr_type = int(parts[1])  # 1-纹理, 2-面料, 3-形状, 4-部件, 5-风格
                attrs.append({
                    'name': attr_name,
                    'type': attr_type
                })
            logger.info(f"读取了 {len(attrs)} 个属性定义")
            return attrs
    except Exception as e:
        logger.error(f"读取属性定义文件失败: {str(e)}")
        return []

def read_attr_img_file(file_path):
    """读取图片属性标注文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是图片数量
            num_imgs = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是图片属性标注
            img_attrs = {}
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                # 将-1转换为0（负例），1保持不变，0表示未知
                attrs = [1 if int(x) == 1 else 0 for x in parts[1:]]
                img_attrs[img_name] = attrs
            logger.info(f"读取了 {len(img_attrs)} 张图片的属性标注")
            return img_attrs
    except Exception as e:
        logger.error(f"读取图片属性标注文件失败: {str(e)}")
        return {}

def read_category_cloth_file(file_path):
    """读取类别定义文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_categories = int(lines[0].strip())
            header = lines[1].strip().split()
            categories = []
            for line in lines[2:]:
                name, type_id = line.strip().split()
                categories.append({
                    'name': name,
                    'type': int(type_id)  # 1-上衣, 2-下装, 3-连衣裙
                })
            logger.info(f"读取了 {len(categories)} 个类别定义")
            return categories
    except Exception as e:
        logger.error(f"读取类别定义文件失败: {str(e)}")
        return []

def read_eval_partition(file_path):
    """读取数据集划分文件"""
    try:
        partition = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            num_imgs = int(lines[0].strip())
            header = lines[1].strip().split()
            for line in lines[2:]:
                img_name, split = line.strip().split()
                partition[img_name] = split
        logger.info(f"读取了 {len(partition)} 张图片的划分信息")
        return partition
    except Exception as e:
        logger.error(f"读取数据集划分文件失败: {str(e)}")
        return {}

def check_image_file(img_path):
    """检查图片文件是否有效"""
    try:
        with Image.open(img_path) as img:
            img.verify()
            return True
    except:
        return False

def merge_attribute_annotations(category_attrs, inshop_attrs, consumer_attrs):
    """合并来自不同数据集的属性标注"""
    # 创建属性映射
    attr_mapping = {}
    for attr in category_attrs:
        name = attr['name']
        if name not in attr_mapping:
            attr_mapping[name] = {
                'type': attr['type'],
                'count': 1,
                'datasets': ['category']
            }
        
    for attr in inshop_attrs:
        name = attr['name']
        if name in attr_mapping:
            attr_mapping[name]['count'] += 1
            attr_mapping[name]['datasets'].append('inshop')
        else:
            attr_mapping[name] = {
                'type': attr.get('type', 0),  # 如果没有类型信息，默认为0
                'count': 1,
                'datasets': ['inshop']
            }
            
    for attr in consumer_attrs:
        name = attr['name']
        if name in attr_mapping:
            attr_mapping[name]['count'] += 1
            attr_mapping[name]['datasets'].append('consumer')
        else:
            attr_mapping[name] = {
                'type': attr.get('type', 0),
                'count': 1,
                'datasets': ['consumer']
            }
    
    # 只保留在至少两个数据集中出现的属性
    common_attrs = {
        name: info for name, info in attr_mapping.items()
        if info['count'] >= 2
    }
    
    logger.info(f"找到 {len(common_attrs)} 个共同属性")
    return common_attrs

def prepare_deepfashion_dataset(min_samples_per_attr=100):
    """准备DeepFashion多数据集"""
    try:
        # 1. 读取各数据集的属性定义
        category_attrs = read_attr_cloth_file(os.path.join(CATEGORY_ANNO_DIR, "list_attr_cloth.txt"))
        inshop_attrs = read_attr_cloth_file(os.path.join(INSHOP_ANNO_DIR, "attributes/list_attr_cloth.txt"))
        consumer_attrs = read_attr_cloth_file(os.path.join(CONSUMER2SHOP_ANNO_DIR, "list_attr_cloth.txt"))
        
        # 2. 合并属性定义
        common_attrs = merge_attribute_annotations(category_attrs, inshop_attrs, consumer_attrs)
        
        # 3. 读取各数据集的图片标注
        category_img_attrs = read_attr_img_file(os.path.join(CATEGORY_ANNO_DIR, "list_attr_img.txt"))
        inshop_img_attrs = read_attr_img_file(os.path.join(INSHOP_ANNO_DIR, "attributes/list_attr_items.txt"))
        consumer_img_attrs = read_attr_img_file(os.path.join(CONSUMER2SHOP_ANNO_DIR, "list_attr_img.txt"))
        
        # 4. 读取数据集划分信息
        category_partition = read_eval_partition(os.path.join(CATEGORY_EVAL_DIR, "list_eval_partition.txt"))
        inshop_partition = read_eval_partition(os.path.join(INSHOP_EVAL_DIR, "list_eval_partition.txt"))
        consumer_partition = read_eval_partition(os.path.join(CONSUMER2SHOP_EVAL_DIR, "list_eval_partition.txt"))
        
        # 5. 合并数据
        data = []
        
        # 处理Category数据集图片
        for img_name, attrs in tqdm(category_img_attrs.items(), desc="处理Category数据集"):
            img_path = os.path.join(CATEGORY_IMG_DIR, img_name)
            if not os.path.exists(img_path) or not check_image_file(img_path):
                continue
                
            split = category_partition.get(img_name)
            if not split:
                continue
                
            data.append({
                'image_path': img_path,
                'split': split,
                'dataset': 'category',
                'attributes': attrs
            })
            
        # 处理In-shop数据集图片
        for img_name, attrs in tqdm(inshop_img_attrs.items(), desc="处理In-shop数据集"):
            img_path = os.path.join(INSHOP_IMG_DIR, img_name)
            if not os.path.exists(img_path) or not check_image_file(img_path):
                continue
                
            split = inshop_partition.get(img_name)
            if not split:
                continue
                
            data.append({
                'image_path': img_path,
                'split': split,
                'dataset': 'inshop',
                'attributes': attrs
            })
            
        # 处理Consumer-to-shop数据集图片
        for img_name, attrs in tqdm(consumer_img_attrs.items(), desc="处理Consumer-to-shop数据集"):
            img_path = os.path.join(CONSUMER2SHOP_IMG_DIR, img_name)
            if not os.path.exists(img_path) or not check_image_file(img_path):
                continue
                
            split = consumer_partition.get(img_name)
            if not split:
                continue
                
            data.append({
                'image_path': img_path,
                'split': split,
                'dataset': 'consumer',
                'attributes': attrs
            })
            
        # 6. 创建DataFrame
        df = pd.DataFrame(data)
        
        # 7. 过滤样本数过少的属性
        valid_attrs = []
        attr_stats = {}
        
        for attr_name in common_attrs:
            # 统计在所有数据集中的正例数量
            pos_samples = sum(1 for d in data if d['attributes'].get(attr_name, 0) == 1)
            attr_stats[attr_name] = {
                'positive_samples': pos_samples,
                'type': common_attrs[attr_name]['type'],
                'datasets': common_attrs[attr_name]['datasets']
            }
            
            if pos_samples >= min_samples_per_attr:
                valid_attrs.append(attr_name)
                logger.info(f"属性 '{attr_name}' 的正例数量: {pos_samples}")
            else:
                logger.warning(f"属性 '{attr_name}' 的正例数量不足 ({pos_samples} < {min_samples_per_attr})")
        
        # 8. 只保留有效属性
        df_filtered = df[['image_path', 'split', 'dataset'] + valid_attrs]
        
        # 9. 分割数据集
        train_df = df_filtered[df_filtered['split'] == 'train'].drop(['split', 'dataset'], axis=1)
        val_df = df_filtered[df_filtered['split'] == 'val'].drop(['split', 'dataset'], axis=1)
        test_df = df_filtered[df_filtered['split'] == 'test'].drop(['split', 'dataset'], axis=1)
        
        # 10. 保存处理后的数据
        output_dir = "data/deepfashion_merged"
        os.makedirs(output_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(output_dir, "train_attr.txt"), sep=' ', index=False)
        val_df.to_csv(os.path.join(output_dir, "val_attr.txt"), sep=' ', index=False)
        test_df.to_csv(os.path.join(output_dir, "test_attr.txt"), sep=' ', index=False)
        
        # 11. 保存属性统计信息
        attr_stats_df = pd.DataFrame([{
            'attribute': attr,
            'type': attr_stats[attr]['type'],
            'positive_samples': attr_stats[attr]['positive_samples'],
            'datasets': ','.join(attr_stats[attr]['datasets'])
        } for attr in valid_attrs])
        
        attr_stats_df.to_csv(os.path.join(output_dir, "attribute_stats.csv"), index=False)
        
        # 12. 输出统计信息
        dataset_stats = df_filtered.groupby(['dataset', 'split']).size().unstack()
        logger.info("\n数据集统计:")
        logger.info(f"\n{dataset_stats}")
        
        logger.info(f"\n总计:")
        logger.info(f"- 训练集: {len(train_df)} 张图片")
        logger.info(f"- 验证集: {len(val_df)} 张图片")
        logger.info(f"- 测试集: {len(test_df)} 张图片")
        logger.info(f"- 有效属性: {len(valid_attrs)}/{len(common_attrs)}")
        
        return len(valid_attrs)
        
    except Exception as e:
        logger.error(f"数据准备失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)
    
    # 准备数据集
    num_attrs = prepare_deepfashion_dataset(min_samples_per_attr=100)
    if num_attrs:
        logger.info(f"成功准备数据集，共有 {num_attrs} 个有效属性") 