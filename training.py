import json
import logging
import os
import copy
import random

import torch
import optuna
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from PIL import Image, ImageFile
from functools import partial
from optuna.trial import Trial
from tqdm import tqdm

from base_model import FullModel, FocalLoss, MultiTaskLoss
from dataset import DeepFashionDataset

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置常量
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

# DeepFashion数据集路径
DEEPFASHION_ROOT = "/home/cv_model/deepfashion"
CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
CATEGORY_IMG_DIR = os.path.join(CATEGORY_ROOT, "Img", "img_highres/img_highres")
CATEGORY_ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")

# Fashion Landmark Detection Benchmark路径
LANDMARK_ROOT = os.path.join(DEEPFASHION_ROOT, "Fashion Landmark Detection Benchmark")
LANDMARK_ANNO_DIR = os.path.join(LANDMARK_ROOT, "Anno")

# 数据集文件
TRAIN_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_attr.txt")
VAL_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_attr.txt")
TEST_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "test_attr.txt")
TRAIN_SEG_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_seg.txt")
VAL_SEG_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_seg.txt")
TEST_SEG_FILE = os.path.join(CATEGORY_ANNO_DIR, "test_seg.txt")

# 关键点标注文件
TRAIN_LANDMARK_FILE = os.path.join(LANDMARK_ANNO_DIR, "list_landmarks.txt")
VAL_LANDMARK_FILE = os.path.join(LANDMARK_ANNO_DIR, "list_landmarks.txt")  # 使用同一个文件，后面根据eval partition拆分
TEST_LANDMARK_FILE = os.path.join(LANDMARK_ANNO_DIR, "list_landmarks.txt")

# 属性和类别定义文件
ATTR_CLOTH_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_attr_cloth.txt")
CATEGORY_CLOTH_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_category_cloth.txt")

# Category and Attribute Prediction Benchmark
CATEGORY_EVAL_DIR = os.path.join(CATEGORY_ROOT, "Eval")

# In-shop Clothes Retrieval Benchmark
INSHOP_ROOT = os.path.join(DEEPFASHION_ROOT, "In-shop Clothes Retrieval Benchmark")
INSHOP_IMG_DIR = os.path.join(INSHOP_ROOT, "Img/img_highres/img_highres")
INSHOP_ANNO_DIR = os.path.join(INSHOP_ROOT, "Anno")
INSHOP_EVAL_DIR = os.path.join(INSHOP_ROOT, "Eval")

# Consumer-to-shop Clothes Retrieval Benchmark
CONSUMER2SHOP_ROOT = os.path.join(DEEPFASHION_ROOT, "Consumer-to-shop Clothes Retrieval Benchmark")
CONSUMER2SHOP_IMG_DIR = os.path.join(CONSUMER2SHOP_ROOT, "Img/img_highres/img_highres")
CONSUMER2SHOP_ANNO_DIR = os.path.join(CONSUMER2SHOP_ROOT, "Anno")
CONSUMER2SHOP_EVAL_DIR = os.path.join(CONSUMER2SHOP_ROOT, "Eval")

# 定义标注文件路径
ATTR_IMG_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_attr_img.txt")
EVAL_PARTITION_FILE = os.path.join(CATEGORY_EVAL_DIR, "list_eval_partition.txt")


# ---------------------- 增强的数据验证类 ----------------------
class DataValidator:
    @staticmethod
    def validate_image(img_path):
        """验证图像文件"""
        full_path = os.path.join(CATEGORY_IMG_DIR, img_path)
        if not os.path.exists(full_path):
            logger.warning(f"图像文件不存在: {full_path}")
            return False

        try:
            with Image.open(full_path) as img:
                # 检查图像是否可以正常打开
                img.verify()
                # 重新打开图像以检查尺寸
                img = Image.open(full_path)
                width, height = img.size
                if width < 10 or height < 10:
                    logger.warning(f"图像尺寸过小: {full_path}, size={img.size}")
                    return False
                return True
        except Exception as e:
            logger.warning(f"图像验证失败: {full_path}, 错误: {str(e)}")
            return False

    @staticmethod
    def validate_labels(labels):
        """验证标签数据"""
        try:
            # 确保所有值都是0或1
            label_values = labels.values
            return np.all(np.logical_or(label_values == 0, label_values == 1))
        except Exception as e:
            logger.warning(f"标签验证失败: {str(e)}")
            return False


# ---------------------- TextileNet数据集类 ----------------------
class TextileNetDataset(Dataset):
    """TextileNet纹理分类数据集（fabric和fiber）"""
    
    def __init__(self, 
                 root_dir: str,  # fabric或fiber根目录
                 dataset_type: str = 'fabric',  # 'fabric' 或 'fiber'
                 split: str = 'train',  # 'train' 或 'test'
                 transform=None,
                 max_retry: int = 5):
        """初始化TextileNet数据集
        
        Args:
            root_dir: 数据集根目录路径
            dataset_type: 数据集类型，'fabric'（面料）或'fiber'（纤维）
            split: 数据集分割，'train'或'test'
            transform: 数据增强和预处理
            max_retry: 加载失败时的最大重试次数
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.split = split
        self.transform = transform
        self.max_retry = max_retry
        
        # 构建数据集路径
        self.data_dir = os.path.join(root_dir, dataset_type, split)
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据集路径不存在: {self.data_dir}")
        
        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(self.data_dir) 
                              if os.path.isdir(os.path.join(self.data_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # 收集所有图像文件
        self.samples = []
        self._load_samples()
        
        logger.info(f"TextileNet {dataset_type} {split}数据集加载完成:")
        logger.info(f"- 类别数: {len(self.classes)}")
        logger.info(f"- 样本数: {len(self.samples)}")
        logger.info(f"- 类别: {self.classes}")
    
    def _load_samples(self):
        """加载所有样本"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # 遍历类别目录中的所有图像文件
            for filename in os.listdir(class_dir):
                if any(filename.lower().endswith(ext) for ext in supported_extensions):
                    img_path = os.path.join(class_dir, filename)
                    self.samples.append((img_path, class_idx, class_name))
    
    def get_class_names(self):
        """获取类别名称列表"""
        return self.classes
    
    def get_num_classes(self):
        """获取类别数量"""
        return len(self.classes)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                img_path, class_idx, class_name = self.samples[idx]
                
                # 读取图片
                image = Image.open(img_path).convert('RGB')
                
                # 数据增强
                if self.transform:
                    image = self.transform(image)
                
                # 准备样本数据
                sample = {
                    'image': image,
                    'img_name': img_path,
                    'class_idx': class_idx,
                    'class_name': class_name,
                    'dataset_type': self.dataset_type
                }
                
                return sample
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"加载TextileNet样本 {idx} 失败 (尝试 {retry_count}/{self.max_retry}): {str(e)}")
                if retry_count == self.max_retry:
                    logger.error(f"无法加载TextileNet样本 {idx}，已达到最大重试次数")
                    raise RuntimeError(f"无法加载TextileNet样本 {idx}")
                # 随机选择另一个样本
                idx = random.randint(0, len(self) - 1)
    
    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.samples)


# ---------------------- 混合数据集类 ----------------------
class MixedDataset(Dataset):
    """混合DeepFashion和TextileNet数据集"""
    
    def __init__(self, 
                 deepfashion_dataset: DeepFashionDataset = None,
                 fabric_dataset: TextileNetDataset = None,
                 fiber_dataset: TextileNetDataset = None,
                 mixing_strategy: str = 'balanced',  # 'balanced', 'weighted', 'sequential'
                 deepfashion_weight: float = 0.5,
                 fabric_weight: float = 0.25,
                 fiber_weight: float = 0.25):
        """初始化混合数据集
        
        Args:
            deepfashion_dataset: DeepFashion数据集
            fabric_dataset: Fabric纹理数据集
            fiber_dataset: Fiber纤维数据集
            mixing_strategy: 混合策略
            deepfashion_weight: DeepFashion数据集权重
            fabric_weight: Fabric数据集权重
            fiber_weight: Fiber数据集权重
        """
        self.datasets = {}
        self.dataset_lengths = {}
        self.dataset_weights = {}
        
        # 添加数据集
        if deepfashion_dataset is not None:
            self.datasets['deepfashion'] = deepfashion_dataset
            self.dataset_lengths['deepfashion'] = len(deepfashion_dataset)
            self.dataset_weights['deepfashion'] = deepfashion_weight
            
        if fabric_dataset is not None:
            self.datasets['fabric'] = fabric_dataset
            self.dataset_lengths['fabric'] = len(fabric_dataset)
            self.dataset_weights['fabric'] = fabric_weight
            
        if fiber_dataset is not None:
            self.datasets['fiber'] = fiber_dataset
            self.dataset_lengths['fiber'] = len(fiber_dataset)
            self.dataset_weights['fiber'] = fiber_weight
        
        if not self.datasets:
            raise ValueError("至少需要提供一个数据集")
        
        self.mixing_strategy = mixing_strategy
        self.total_length = self._calculate_total_length()
        
        # 创建采样索引
        self.sample_indices = self._create_sample_indices()
        
        logger.info(f"混合数据集创建完成:")
        logger.info(f"- 混合策略: {mixing_strategy}")
        logger.info(f"- 数据集: {list(self.datasets.keys())}")
        logger.info(f"- 各数据集长度: {self.dataset_lengths}")
        logger.info(f"- 各数据集权重: {self.dataset_weights}")
        logger.info(f"- 总长度: {self.total_length}")
    
    def _calculate_total_length(self):
        """计算总长度"""
        if self.mixing_strategy == 'balanced':
            # 平衡策略：使用最大数据集的长度
            return max(self.dataset_lengths.values())
        elif self.mixing_strategy == 'weighted':
            # 加权策略：根据权重计算总长度
            return int(sum(length * weight for length, weight in 
                          zip(self.dataset_lengths.values(), self.dataset_weights.values())))
        elif self.mixing_strategy == 'sequential':
            # 顺序策略：所有数据集长度之和
            return sum(self.dataset_lengths.values())
        else:
            raise ValueError(f"不支持的混合策略: {self.mixing_strategy}")
    
    def _create_sample_indices(self):
        """创建采样索引"""
        indices = []
        
        if self.mixing_strategy == 'balanced':
            # 平衡采样：每个数据集按比例采样
            for i in range(self.total_length):
                # 根据权重随机选择数据集
                dataset_probs = list(self.dataset_weights.values())
                dataset_names = list(self.datasets.keys())
                chosen_dataset = np.random.choice(dataset_names, p=dataset_probs)
                
                # 在选中的数据集中随机选择样本
                dataset_idx = np.random.randint(0, self.dataset_lengths[chosen_dataset])
                indices.append((chosen_dataset, dataset_idx))
                
        elif self.mixing_strategy == 'weighted':
            # 加权采样
            for dataset_name, weight in self.dataset_weights.items():
                num_samples = int(self.dataset_lengths[dataset_name] * weight)
                for i in range(num_samples):
                    dataset_idx = i % self.dataset_lengths[dataset_name]
                    indices.append((dataset_name, dataset_idx))
            
        elif self.mixing_strategy == 'sequential':
            # 顺序采样
            for dataset_name in self.datasets.keys():
                for i in range(self.dataset_lengths[dataset_name]):
                    indices.append((dataset_name, i))
        
        # 打乱索引
        np.random.shuffle(indices)
        return indices
    
    def __getitem__(self, idx):
        """获取单个样本"""
        dataset_name, dataset_idx = self.sample_indices[idx]
        dataset = self.datasets[dataset_name]
        
        sample = dataset[dataset_idx]
        
        # 添加数据集来源信息
        sample['source_dataset'] = dataset_name
        
        # 根据数据集类型处理标签
        if dataset_name == 'deepfashion':
            # DeepFashion保持原有的attr_labels
            pass
        else:
            # TextileNet数据集：将类别索引转换为属性标签格式
            # 这里需要根据具体需求设计映射策略
            num_attrs = 26  # DeepFashion的属性数量
            attr_labels = torch.zeros(num_attrs, dtype=torch.float32)
            
            # 简单映射策略：将纹理类别映射到特定属性位置
            if dataset_name == 'fabric':
                # 面料类别映射到纹理相关属性
                attr_labels[0] = 1.0  # 假设第0个属性是纹理相关
            elif dataset_name == 'fiber':
                # 纤维类别映射到材质相关属性
                attr_labels[1] = 1.0  # 假设第1个属性是材质相关
            
            sample['attr_labels'] = attr_labels
            sample['textile_class_idx'] = sample['class_idx']
            sample['textile_class_name'] = sample['class_name']
        
        return sample
    
    def __len__(self):
        """返回数据集的总样本数"""
        return self.total_length
    
    def get_dataset_info(self):
        """获取数据集信息"""
        info = {
            'total_length': self.total_length,
            'datasets': {},
            'mixing_strategy': self.mixing_strategy
        }
        
        for name, dataset in self.datasets.items():
            if name == 'deepfashion':
                info['datasets'][name] = {
                    'length': len(dataset),
                    'weight': self.dataset_weights[name],
                    'attr_names': dataset.get_attr_names() if hasattr(dataset, 'get_attr_names') else None
                }
            else:
                info['datasets'][name] = {
                    'length': len(dataset),
                    'weight': self.dataset_weights[name],
                    'classes': dataset.get_class_names(),
                    'num_classes': dataset.get_num_classes()
                }
        
        return info


# ---------------------- 改进的数据集类 ----------------------
class DeepFashionDataset(Dataset):
    """DeepFashion多任务数据集"""

    def __init__(self,
                 img_list_file: str,  # 图片列表文件
                 attr_file: str,  # 属性标注文件
                 cate_file: str = None,  # 类别标注文件
                 bbox_file: str = None,  # 边界框标注文件
                 seg_file: str = None,  # 分割标注文件
                 image_dir: str = None,  # 图像根目录
                 transform=None,
                 max_retry: int = 5):
        """初始化数据集

        Args:
            img_list_file: 图片列表文件路径
            attr_file: 属性标注文件路径
            cate_file: 类别标注文件路径（可选）
            bbox_file: 边界框标注文件路径（可选）
            seg_file: 分割标注文件路径（可选）
            image_dir: 图像根目录路径（可选）
            transform: 数据增强和预处理
            max_retry: 加载失败时的最大重试次数
        """
        self.transform = transform
        self.image_dir = image_dir
        self.max_retry = max_retry
        self.attr_file = attr_file  # 保存属性文件路径

        # 读取图片列表
        try:
            with open(img_list_file, 'r', encoding='utf-8') as f:
                self.img_paths = [line.strip() for line in f]
            logger.info(f"读取了 {len(self.img_paths)} 个图片路径")
        except Exception as e:
            logger.error(f"读取图片列表文件失败: {str(e)}")
            raise

        # 读取属性标注
        try:
            data = []
            with open(attr_file, 'r', encoding='utf-8') as f:
                for line in f:
                    attrs = [int(x) for x in line.strip().split()]
                    data.append(attrs)

            self.attr_labels = torch.tensor(data, dtype=torch.float32)
            logger.info(f"读取了 {len(self.attr_labels)} 个属性标注")
        except Exception as e:
            logger.error(f"读取属性标注文件失败: {str(e)}")
            raise

        # 读取类别标注（如果有）
        self.cate_labels = None
        if cate_file and os.path.exists(cate_file):
            try:
                categories = []
                with open(cate_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        cate = int(line.strip())
                        categories.append(cate)
                self.cate_labels = torch.tensor(categories, dtype=torch.long)
                logger.info(f"读取了 {len(self.cate_labels)} 个类别标注")
            except Exception as e:
                logger.error(f"读取类别标注文件失败: {str(e)}")

        # 读取边界框标注（如果有）
        self.bbox_labels = None
        if bbox_file and os.path.exists(bbox_file):
            try:
                bboxes = []
                with open(bbox_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # x1, y1, x2, y2
                        bbox = [float(x) for x in line.strip().split()]
                        bboxes.append(bbox)
                self.bbox_labels = torch.tensor(bboxes, dtype=torch.float32)
                logger.info(f"读取了 {len(self.bbox_labels)} 个边界框标注")
            except Exception as e:
                logger.error(f"读取边界框标注文件失败: {str(e)}")

        # 读取分割标注（如果有）
        self.seg_labels = None
        if seg_file and os.path.exists(seg_file):
            try:
                segmentations = []
                with open(seg_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # 假设分割标注是二值图像，展平为一维向量
                        seg = [float(x) for x in line.strip().split()]
                        segmentations.append(seg)
                self.seg_labels = torch.tensor(segmentations, dtype=torch.float32)
                logger.info(f"读取了 {len(self.seg_labels)} 个分割标注")
            except Exception as e:
                logger.error(f"读取分割标注文件失败: {str(e)}")

        # 验证数据一致性
        self._validate_data()

    def get_attr_names(self):
        """获取属性名称列表"""
        try:
            # 从属性文件路径推断属性定义文件路径
            attr_dir = os.path.dirname(self.attr_file)
            attr_cloth_file = os.path.join(attr_dir, "list_attr_cloth.txt")

            if not os.path.exists(attr_cloth_file):
                logger.warning(f"找不到属性定义文件: {attr_cloth_file}")
                # 如果找不到属性定义文件，返回数字索引作为属性名称
                return [f"attr_{i}" for i in range(self.attr_labels.shape[1])]

            # 读取属性定义文件
            with open(attr_cloth_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 跳过前两行（属性数量和表头）
                attr_names = [line.strip().split()[0] for line in lines[2:]]

            logger.info(f"读取了 {len(attr_names)} 个属性名称")
            return attr_names

        except Exception as e:
            logger.error(f"读取属性名称失败: {str(e)}")
            # 发生错误时返回数字索引作为属性名称
            return [f"attr_{i}" for i in range(self.attr_labels.shape[1])]

    def _validate_data(self):
        """验证所有标注数据的样本数量是否一致"""
        num_samples = len(self.img_paths)

        if len(self.attr_labels) != num_samples:
            raise ValueError(f"属性标注数量({len(self.attr_labels)})与图片数量({num_samples})不匹配")

        if self.cate_labels is not None and len(self.cate_labels) != num_samples:
            raise ValueError(f"类别标注数量({len(self.cate_labels)})与图片数量({num_samples})不匹配")

        if self.bbox_labels is not None and len(self.bbox_labels) != num_samples:
            raise ValueError(f"边界框标注数量({len(self.bbox_labels)})与图片数量({num_samples})不匹配")

        if self.seg_labels is not None and len(self.seg_labels) != num_samples:
            raise ValueError(f"分割标注数量({len(self.seg_labels)})与图片数量({num_samples})不匹配")

    def __getitem__(self, idx):
        """获取单个样本"""
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                # 获取图片路径
                img_path = self.img_paths[idx]
                if self.image_dir:
                    img_path = os.path.join(self.image_dir, img_path)

                # 获取图像名称（包含相对路径）
                img_name = img_path

                # 读取图片
                image = Image.open(img_path).convert('RGB')

                # 数据增强
                if self.transform:
                    image = self.transform(image)

                # 准备样本数据
                sample = {
                    'image': image,
                    'img_name': img_name,
                    'attr_labels': self.attr_labels[idx],
                }

                # 添加类别标签（如果有）
                if self.cate_labels is not None:
                    sample['category_labels'] = self.cate_labels[idx]

                # 添加边界框标签（如果有）
                if self.bbox_labels is not None:
                    sample['bbox_labels'] = self.bbox_labels[idx]

                # 添加分割标签（如果有）
                if self.seg_labels is not None:
                    sample['segmentation'] = self.seg_labels[idx]

                return sample

            except Exception as e:
                retry_count += 1
                logger.warning(f"加载样本 {idx} 失败 (尝试 {retry_count}/{self.max_retry}): {str(e)}")
                if retry_count == self.max_retry:
                    logger.error(f"无法加载样本 {idx}，已达到最大重试次数")
                    raise RuntimeError(f"无法加载样本 {idx}")
                # 随机选择另一个样本
                idx = random.randint(0, len(self) - 1)

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.img_paths)


# ---------------------- 增强的训练模块 ----------------------
class MixedDatasetTrainer:
    """混合数据集多任务训练器"""

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 3e-4,
                 enable_segmentation: bool = True,
                 enable_textile_classification: bool = True):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.enable_segmentation = enable_segmentation
        self.enable_textile_classification = enable_textile_classification

        # 优化器配置
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.15
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=50,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )

        # 损失函数
        self.attr_criterion = FocalLoss(gamma=4.0, alpha=0.5)
        self.segmentation_criterion = nn.BCEWithLogitsLoss()
        self.textile_criterion = nn.CrossEntropyLoss()  # 纹理分类损失

        # 多任务损失
        num_tasks = 1 + int(enable_segmentation) + int(enable_textile_classification)
        self.multi_task_loss = MultiTaskLoss(
            num_tasks=num_tasks,
            device=device
        )

        # EMA模型
        self.ema = torch.optim.swa_utils.AveragedModel(model)
        self.ema_start = 20

        # 梯度缩放器
        self.scaler = torch.amp.GradScaler('cuda')

        # 早停
        self.patience = 15
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # 当前epoch
        self.current_epoch = 0

        # 多GPU支持
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU!")

    def _train_step(self, batch):
        """执行单个训练步骤

        Args:
            batch: 包含图像和标签的批次数据

        Returns:
            dict: 包含各项损失的字典
        """
        # 将数据移到设备上
        images = batch['image'].to(self.device)
        attr_labels = batch['attr_labels'].to(self.device)
        img_names = batch['img_name']  # 获取图像名称

        # 获取分割任务的标签（如果有）
        seg_labels = batch.get('segmentation')
        if seg_labels is not None:
            seg_labels = seg_labels.to(self.device)

        # 获取纹理分类标签（如果有）
        textile_labels = batch.get('textile_class_idx')
        if textile_labels is not None:
            textile_labels = torch.stack([torch.tensor(label) if not isinstance(label, torch.Tensor) 
                                        else label for label in textile_labels]).to(self.device)

        # 获取数据集来源信息
        source_datasets = batch.get('source_dataset', ['deepfashion'] * len(images))

        # 清零梯度
        self.optimizer.zero_grad()

        # 使用AMP进行前向传播
        with torch.amp.autocast('cuda'):
            outputs = self.model(images, img_names)  # 传递图像名称

            # 计算各任务的损失
            losses = {}
            loss_list = []

            # 1. 属性分类损失（所有数据集都计算）
            attr_loss = self.attr_criterion(outputs['attr_logits'], attr_labels)
            losses['attr_loss'] = attr_loss
            loss_list.append(attr_loss)

            # 2. 分割损失（如果启用且有标签）
            if self.enable_segmentation and seg_labels is not None:
                seg_loss = self.segmentation_criterion(outputs['seg_logits'], seg_labels)
                losses['seg_loss'] = seg_loss
                loss_list.append(seg_loss)
            else:
                losses['seg_loss'] = torch.tensor(0.0, device=self.device)
                if self.enable_segmentation:
                    loss_list.append(torch.tensor(0.0, device=self.device))

            # 3. 纹理分类损失（仅对TextileNet数据计算）
            if self.enable_textile_classification and textile_labels is not None:
                # 检查是否有来自TextileNet的样本
                textile_mask = torch.tensor([ds in ['fabric', 'fiber'] for ds in source_datasets], 
                                          device=self.device)
                
                if textile_mask.any():
                    # 获取纹理分类输出（需要在模型中添加）
                    if 'textile_logits' in outputs:
                        textile_indices = torch.where(textile_mask)[0]
                        textile_pred = outputs['textile_logits'][textile_indices]
                        textile_true = textile_labels[textile_indices]
                        textile_loss = self.textile_criterion(textile_pred, textile_true)
                        losses['textile_loss'] = textile_loss
                        loss_list.append(textile_loss)
                    else:
                        losses['textile_loss'] = torch.tensor(0.0, device=self.device)
                        loss_list.append(torch.tensor(0.0, device=self.device))
                else:
                    losses['textile_loss'] = torch.tensor(0.0, device=self.device)
                    loss_list.append(torch.tensor(0.0, device=self.device))
            else:
                losses['textile_loss'] = torch.tensor(0.0, device=self.device)
                if self.enable_textile_classification:
                    loss_list.append(torch.tensor(0.0, device=self.device))

            # 使用多任务损失进行加权
            if len(loss_list) > 1:
                total_loss = self.multi_task_loss(loss_list)
            else:
                total_loss = loss_list[0]
            
            losses['total_loss'] = total_loss

        # 反向传播
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 更新学习率
        self.scheduler.step()

        # 更新EMA模型
        if self.current_epoch >= self.ema_start:
            self.ema.update_parameters(self.model)

        return losses

    def train_epoch(self):
        self.model.train()
        metrics = {
            'total_loss': 0.0,
            'attr_loss': 0.0,
            'seg_loss': 0.0,
            'textile_loss': 0.0
        }

        # 添加进度条
        pbar = tqdm(self.train_loader)
        for batch in pbar:
            # 训练步骤
            loss = self._train_step(batch)

            # 更新指标
            for k, v in loss.items():
                metrics[k] += v

            # 更新进度条
            pbar.set_postfix({k: f'{v / len(pbar):.4f}' for k, v in metrics.items()})

        return {k: v / len(self.train_loader) for k, v in metrics.items()}

    def validate(self):
        # 使用EMA模型进行验证
        if self.current_epoch >= self.ema_start:
            eval_model = self.ema.module
        else:
            eval_model = self.model

        eval_model.eval()
        val_loss = 0.0
        attr_correct = 0
        attr_total = 0
        seg_iou = 0.0
        seg_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                attr_labels = batch['attr_labels'].to(self.device)
                img_names = batch['img_name']  # 获取图像名称

                # 获取分割任务的标签
                seg_labels = batch.get('segmentation')
                if seg_labels is not None:
                    seg_labels = seg_labels.to(self.device)

                # 前向传播
                outputs = eval_model(images, img_names)  # 传递图像名称

                # 1. 属性分类评估
                attr_preds = (outputs['attr_logits'] > 0.5).float()
                attr_correct += (attr_preds == attr_labels).float().sum().item()
                attr_total += attr_labels.numel()

                # 2. 分割评估
                if self.enable_segmentation and seg_labels is not None:
                    seg_preds = (outputs['seg_logits'] > 0).float()
                    intersection = (seg_preds * seg_labels).sum().item()
                    union = (seg_preds + seg_labels).gt(0).sum().item()
                    if union > 0:
                        seg_iou += intersection / union
                        seg_count += 1

        # 计算各项指标
        attr_acc = attr_correct / attr_total if attr_total > 0 else 0
        seg_avg_iou = seg_iou / seg_count if seg_count > 0 else 0

        return {
            'attr_acc': attr_acc,
            'seg_iou': seg_avg_iou
        }

    def train(self, epochs: int = 50, save_dir: str = "checkpoints"):
        """训练模型
        Args:
            epochs: 训练轮数
            save_dir: 模型保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 更新scheduler的epochs参数
        self.scheduler.total_steps = epochs * len(self.train_loader)

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 输出训练信息
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("训练指标:")
            for k, v in train_metrics.items():
                logger.info(f"- {k}: {v:.4f}")
            logger.info("\n验证指标:")
            for k, v in val_metrics.items():
                logger.info(f"- {k}: {v:.4f}")

            # 早停检查
            val_loss = -val_metrics['attr_acc']  # 使用属性准确率的负值作为验证损失
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型（完整模型）
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model, model_path)
                logger.info(f"保存最佳模型到: {model_path}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                logger.info(f"在第 {epoch + 1} 轮提前停止训练")
                break

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }

        # 保存最新checkpoint
        torch.save(state, 'checkpoint_latest.pth')

        # 如果是最佳模型,额外保存
        if is_best:
            torch.save(state, 'checkpoint_best.pth')


# ---------------------- 改进的数据增强配置 ----------------------
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


# ---------------------- 数据加载工具函数 ----------------------
def collate_fn(batch):
    """处理批量数据的组合函数"""
    # 过滤掉None（无效样本）
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    # 组合所有样本
    output = {}
    for key in batch[0].keys():
        if key == 'img_name':
            output[key] = [b[key] for b in batch]
        elif isinstance(batch[0][key], torch.Tensor):
            output[key] = torch.stack([b[key] for b in batch])
        else:
            output[key] = [b[key] for b in batch]

    return output


# ---------------------- 改进的DataLoader配置 ----------------------
def create_safe_loader(dataset, batch_size=32, num_workers=4, pin_memory=True):
    """创建带有异常处理的DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn
    )


# ---------------------- 改进的K折交叉验证 ----------------------
class KFoldTrainer:
    """增强的K折交叉验证训练器"""

    def __init__(self,
                 dataset: Dataset,
                 num_folds: int = 5,
                 batch_size: int = 32,
                 epochs_per_fold: int = 40,
                 save_dir: str = 'checkpoints'):
        self.dataset = dataset
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.epochs_per_fold = epochs_per_fold
        self.save_dir = save_dir
        self.kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)

        # 存储各折结果
        self.fold_results = []
        self.best_models = []

    def run(self):
        """执行交叉验证"""
        overall_best_acc = 0

        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.dataset)):
            logger.info(f"\n{'=' * 30} Fold {fold + 1}/{self.num_folds} {'=' * 30}")

            # 创建采样器
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            # 创建数据加载器
            train_loader = create_safe_loader(
                self.dataset,
                batch_size=self.batch_size,
            )
            val_loader = create_safe_loader(
                self.dataset,
                batch_size=self.batch_size,
            )

            # 初始化新模型
            model = FullModel()
            trainer = DeepFashionTrainer(model, train_loader, val_loader)

            # 训练当前折
            fold_result = {
                'train_loss': [],
                'val_loss': [],
                'val_acc': []
            }

            best_fold_acc = 0
            best_fold_model = None

            for epoch in range(self.epochs_per_fold):
                train_loss = trainer.train_epoch()
                val_loss, val_acc = trainer.validate()

                fold_result['train_loss'].append(train_loss['total_loss'])
                fold_result['val_loss'].append(val_loss['attr_acc'])
                fold_result['val_acc'].append(val_acc['attr_acc'])

                # 保存最佳模型
                if val_acc['attr_acc'] > best_fold_acc:
                    best_fold_acc = val_acc['attr_acc']
                    best_fold_model = copy.deepcopy(model.state_dict())

                    # 如果是总体最佳，则额外保存
                    if val_acc['attr_acc'] > overall_best_acc:
                        overall_best_acc = val_acc['attr_acc']
                        torch.save(
                            model,  # 保存完整模型
                            os.path.join(self.save_dir, f'best_model_overall.pth')
                        )

                logger.info(
                    f"Fold {fold + 1} Epoch {epoch + 1}/{self.epochs_per_fold} | "
                    f"Train Loss: {train_loss['total_loss']:.4f} | Val Loss: {val_loss['attr_acc']:.4f} | "
                    f"Val Acc: {val_acc['attr_acc']:.4f} | Best Acc: {best_fold_acc:.4f}"
                )

                # 早停检查
                if trainer.patience_counter >= trainer.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            # 保存当前折的最佳模型（完整模型）
            model.load_state_dict(best_fold_model)
            torch.save(
                model,
                os.path.join(self.save_dir, f'best_model_fold_{fold + 1}.pth')
            )

            self.fold_results.append(fold_result)
            self.best_models.append(best_fold_model)

            # 清理GPU内存
            del model, trainer
            torch.cuda.empty_cache()

        self._analyze_results()

    def _analyze_results(self):
        """分析交叉验证结果"""
        # 计算各折最佳性能
        best_accs = [max(fold['val_acc']) for fold in self.fold_results]
        mean_best_acc = np.mean(best_accs)
        std_best_acc = np.std(best_accs)

        # 计算最终轮次的平均性能
        final_accs = [fold['val_acc'][-1] for fold in self.fold_results]
        mean_final_acc = np.mean(final_accs)
        std_final_acc = np.std(final_accs)

        logger.info("\n=== Cross Validation Summary ===")
        logger.info(f"Best Accuracy per Fold: {best_accs}")
        logger.info(f"Mean Best Accuracy: {mean_best_acc:.4f} ± {std_best_acc:.4f}")
        logger.info(f"Mean Final Accuracy: {mean_final_acc:.4f} ± {std_final_acc:.4f}")

        # 可视化训练曲线
        self._plot_learning_curves()

    def _plot_learning_curves(self):
        """绘制详细的学习曲线"""
        plt.figure(figsize=(15, 5))

        # 1. 训练损失
        plt.subplot(1, 3, 1)
        for i, fold in enumerate(self.fold_results):
            plt.plot(fold['train_loss'], label=f'Fold {i + 1}')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 2. 验证损失
        plt.subplot(1, 3, 2)
        for i, fold in enumerate(self.fold_results):
            plt.plot(fold['val_loss'], label=f'Fold {i + 1}')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 3. 验证准确率
        plt.subplot(1, 3, 3)
        for i, fold in enumerate(self.fold_results):
            plt.plot(fold['val_acc'], label=f'Fold {i + 1}')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('cross_validation_results.png')
        plt.close()

    def get_ensemble_model(self):
        """返回集成模型"""
        return ModelEnsemble([
            FullModel().load_state_dict(state_dict)
            for state_dict in self.best_models
        ])


class ModelEnsemble(nn.Module):
    """模型集成类"""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # 对所有模型的预测取平均
        outputs = [model(x) for model in self.models]
        return torch.stack(outputs).mean(dim=0)


class HyperParameterOptimizer:
    """超参数优化器"""

    def __init__(self,
                 dataset: Dataset,
                 num_trials: int = 50,
                 k_folds: int = 3,
                 epochs_per_trial: int = 10):
        self.dataset = dataset
        self.num_trials = num_trials
        self.k_folds = k_folds
        self.epochs_per_trial = epochs_per_trial
        self.study = optuna.create_study(direction='minimize')
        self.best_params = None

    def _objective(self, trial: Trial, model_cls) -> float:
        """定义优化目标函数"""
        # 超参数搜索空间
        params = {
            'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            'gat_heads': trial.suggest_int('gat_heads', 4, 16, step=2),
            'gat_layers': trial.suggest_int('gat_layers', 1, 3),
            'scheduler_gamma': trial.suggest_float('scheduler_gamma', 0.8, 0.95)
        }

        # K折交叉验证
        kfold_trainer = KFoldTrainer(
            dataset=self.dataset,
            num_folds=self.k_folds,
            batch_size=params['batch_size'],
            epochs_per_fold=self.epochs_per_trial
        )

        # 自定义模型初始化
        def create_model():
            return model_cls(
                gat_dims=[512] * params['gat_layers'],
                gat_heads=params['gat_heads']
            )

        # 修改训练器以使用当前超参数
        original_trainer_init = DeepFashionTrainer.__init__

        def patched_trainer_init(self, model, train_loader, val_loader):
            original_trainer_init(self, model, train_loader, val_loader)
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params['lr'],
                weight_decay=params['weight_decay']
            )
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=params['scheduler_gamma']
            )

        # 执行交叉验证
        try:
            DeepFashionTrainer.__init__ = patched_trainer_init
            kfold_trainer.run()
            avg_val_loss = sum(
                [fold['val_loss'][-1] for fold in kfold_trainer.fold_results]) / self.k_folds
        finally:
            DeepFashionTrainer.__init__ = original_trainer_init  # 恢复原始初始化

        return avg_val_loss

    def optimize(self, model_cls=FullModel):
        """执行优化流程"""
        study = optuna.create_study(direction='minimize')
        objective_func = partial(self._objective, model_cls=model_cls)
        study.optimize(objective_func, n_trials=self.num_trials)

        # 保存最佳参数
        self.best_params = study.best_params
        self._save_optimization_result(study)

        return study.best_params

    def _save_optimization_result(self, study):
        """保存优化结果"""
        # 可视化参数重要性
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("param_importance.png")

        # 保存最佳参数
        with open("best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=2)

        # 打印结果
        logger.info(f"Best trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value:.4f}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")


# ---------------------- 混合数据集训练示例 ----------------------
if __name__ == "__main__":
    # 1. 定义DeepFashion文件路径
    DEEPFASHION_ROOT = "/home/cv_model/DeepFashion"
    CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
    ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")
    IMG_DIR = os.path.join(CATEGORY_ROOT, "Img", "img")

    # DeepFashion训练集文件
    TRAIN_IMG_LIST = os.path.join(ANNO_DIR, "train.txt")
    TRAIN_ATTR_FILE = os.path.join(ANNO_DIR, "train_attr.txt")
    TRAIN_CATE_FILE = os.path.join(ANNO_DIR, "train_cate.txt")
    TRAIN_BBOX_FILE = os.path.join(ANNO_DIR, "train_bbox.txt")

    # DeepFashion验证集文件
    VAL_IMG_LIST = os.path.join(ANNO_DIR, "val.txt")
    VAL_ATTR_FILE = os.path.join(ANNO_DIR, "val_attr.txt")
    VAL_CATE_FILE = os.path.join(ANNO_DIR, "val_cate.txt")
    VAL_BBOX_FILE = os.path.join(ANNO_DIR, "val_bbox.txt")
    
    # 2. 定义TextileNet文件路径
    TEXTILE_ROOT = "/home/cv_model"
    FABRIC_ROOT = os.path.join(TEXTILE_ROOT, "fabric")
    FIBER_ROOT = os.path.join(TEXTILE_ROOT, "fiber")

    # 3. 数据增强和预处理
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

    # 4. 创建DeepFashion数据集
    deepfashion_train = DeepFashionDataset(
        img_list_file=TRAIN_IMG_LIST,
        attr_file=TRAIN_ATTR_FILE,
        cate_file=TRAIN_CATE_FILE,
        bbox_file=TRAIN_BBOX_FILE,
        image_dir=IMG_DIR,
        transform=train_transform
    )

    deepfashion_val = DeepFashionDataset(
        img_list_file=VAL_IMG_LIST,
        attr_file=VAL_ATTR_FILE,
        cate_file=VAL_CATE_FILE,
        bbox_file=VAL_BBOX_FILE,
        image_dir=IMG_DIR,
        transform=val_transform
    )
    
    # 5. 创建TextileNet数据集
    fabric_train = TextileNetDataset(
        root_dir=TEXTILE_ROOT,
        dataset_type='fabric',
        split='train',
        transform=train_transform
    )
    
    fabric_val = TextileNetDataset(
        root_dir=TEXTILE_ROOT,
        dataset_type='fabric',
        split='test',  # 使用test作为验证集
        transform=val_transform
    )
    
    fiber_train = TextileNetDataset(
        root_dir=TEXTILE_ROOT,
        dataset_type='fiber',
        split='train',
        transform=train_transform
    )
    
    fiber_val = TextileNetDataset(
        root_dir=TEXTILE_ROOT,
        dataset_type='fiber',
        split='test',  # 使用test作为验证集
        transform=val_transform
    )
    
    # 6. 创建混合数据集
    mixed_train_dataset = MixedDataset(
        deepfashion_dataset=deepfashion_train,
        fabric_dataset=fabric_train,
        fiber_dataset=fiber_train,
        mixing_strategy='balanced',
        deepfashion_weight=0.6,
        fabric_weight=0.2,
        fiber_weight=0.2
    )
    
    mixed_val_dataset = MixedDataset(
        deepfashion_dataset=deepfashion_val,
        fabric_dataset=fabric_val,
        fiber_dataset=fiber_val,
        mixing_strategy='balanced',
        deepfashion_weight=0.6,
        fabric_weight=0.2,
        fiber_weight=0.2
    )
    
    # 打印数据集信息
    logger.info("混合训练数据集信息:")
    train_info = mixed_train_dataset.get_dataset_info()
    for dataset_name, info in train_info['datasets'].items():
        logger.info(f"- {dataset_name}: {info}")
    
    logger.info("混合验证数据集信息:")
    val_info = mixed_val_dataset.get_dataset_info()
    for dataset_name, info in val_info['datasets'].items():
        logger.info(f"- {dataset_name}: {info}")

    # 7. 创建数据加载器
    train_loader = DataLoader(
        mixed_train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        mixed_val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 8. 创建增强模型
    # 获取纹理类别数量
    num_fabric_classes = fabric_train.get_num_classes()
    num_fiber_classes = fiber_train.get_num_classes()
    
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

    # 9. 创建混合数据集训练器
    trainer = MixedDatasetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        enable_segmentation=False,
        enable_textile_classification=True,
        learning_rate=1e-4
    )

    # 10. 开始混合训练
    logger.info("开始混合数据集训练...")
    trainer.train(epochs=50, save_dir="mixed_checkpoints")

    logger.info("混合数据集训练完成！")
