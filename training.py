import json
import logging
import os
import copy

import torch
import optuna
import numpy as np
import pandas as pd
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

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置常量
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像

# DeepFashion数据集路径
DEEPFASHION_ROOT = "deepfashion"
CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
CATEGORY_IMG_DIR = os.path.join(CATEGORY_ROOT, "Img", "img")
CATEGORY_ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")

# 数据集文件
TRAIN_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_attr.txt")
VAL_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_attr.txt")
TEST_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "test_attr.txt")

TRAIN_LANDMARK_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_landmarks.txt")
VAL_LANDMARK_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_landmarks.txt")
TEST_LANDMARK_FILE = os.path.join(CATEGORY_ANNO_DIR, "test_landmarks.txt")

# 属性和类别定义文件
ATTR_CLOTH_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_attr_cloth.txt")
CATEGORY_CLOTH_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_category_cloth.txt")

# Category and Attribute Prediction Benchmark
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

# Fashion Landmark Detection Benchmark
LANDMARK_ROOT = os.path.join(DEEPFASHION_ROOT, "Fashion Landmark Detection Benchmark")
LANDMARK_IMG_DIR = os.path.join(LANDMARK_ROOT, "Img")
LANDMARK_ANNO_DIR = os.path.join(LANDMARK_ROOT, "Anno")

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


# ---------------------- 改进的数据集类 ----------------------
class DeepFashionDataset(Dataset):
    """DeepFashion多任务数据集"""
    def __init__(self,
                 img_list_file: str,  # 图片列表文件
                 attr_file: str,      # 属性标注文件
                 cate_file: str = None,    # 类别标注文件
                 bbox_file: str = None,    # 边界框标注文件
                 landmark_file: str = None, # 关键点标注文件
                 image_dir: str = None,     # 图像根目录
                 transform=None,
                 max_retry: int = 5):
        """
        初始化数据集
        Args:
            img_list_file: 图片列表文件路径（train.txt/val.txt/test.txt）
            attr_file: 属性标注文件路径（train_attr.txt等）
            cate_file: 类别标注文件路径（train_cate.txt等）
            bbox_file: 边界框标注文件路径（train_bbox.txt等）
            landmark_file: 关键点标注文件路径（train_landmark.txt等）
            image_dir: 图像根目录
            transform: 数据增强转换
            max_retry: 最大重试次数
        """
        super().__init__()
        self.transform = transform
        self.max_retry = max_retry
        self.image_dir = image_dir
        
        # 读取图片列表
        try:
            with open(img_list_file, 'r', encoding='utf-8') as f:
                self.img_paths = [line.strip() for line in f.readlines()]
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
                
        # 读取关键点标注（如果有）
        self.landmark_labels = None
        if landmark_file and os.path.exists(landmark_file):
            try:
                landmarks = []
                with open(landmark_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        # x, y, visibility for each landmark
                        coords = [float(x) for x in line.strip().split()]
                        landmarks.append(coords)
                self.landmark_labels = torch.tensor(landmarks, dtype=torch.float32)
                logger.info(f"读取了 {len(self.landmark_labels)} 个关键点标注")
            except Exception as e:
                logger.error(f"读取关键点标注文件失败: {str(e)}")
                
        # 验证数据一致性
        self._validate_data()
        
    def _validate_data(self):
        """验证所有标注数据的样本数量是否一致"""
        num_samples = len(self.img_paths)
        
        if len(self.attr_labels) != num_samples:
            raise ValueError(f"属性标注数量({len(self.attr_labels)})与图片数量({num_samples})不匹配")
            
        if self.cate_labels is not None and len(self.cate_labels) != num_samples:
            raise ValueError(f"类别标注数量({len(self.cate_labels)})与图片数量({num_samples})不匹配")
            
        if self.bbox_labels is not None and len(self.bbox_labels) != num_samples:
            raise ValueError(f"边界框标注数量({len(self.bbox_labels)})与图片数量({num_samples})不匹配")
            
        if self.landmark_labels is not None and len(self.landmark_labels) != num_samples:
            raise ValueError(f"关键点标注数量({len(self.landmark_labels)})与图片数量({num_samples})不匹配")
        
    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        for _ in range(self.max_retry):
            try:
                # 获取图像路径（图片列表中的路径已经包含了相对路径）
                img_path = os.path.join(self.image_dir, self.img_paths[idx])
                
                # 读取图像
                if os.path.exists(img_path):
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                else:
                    logger.warning(f"图像文件不存在: {img_path}")
                    idx = (idx + 1) % len(self.img_paths)
                    continue
                
                # 准备输出数据
                output = {
                    'image': image,
                    'attr_labels': self.attr_labels[idx]
                }
                
                # 添加类别标注（如果有）
                if self.cate_labels is not None:
                    output['category'] = self.cate_labels[idx]
                    
                # 添加边界框标注（如果有）
                if self.bbox_labels is not None:
                    output['bbox'] = self.bbox_labels[idx]
                    
                # 添加关键点标注（如果有）
                if self.landmark_labels is not None:
                    landmarks = self.landmark_labels[idx].reshape(-1, 3)  # reshape to (num_landmarks, 3)
                    output['landmarks'] = landmarks
                    
                return output
                    
            except Exception as e:
                logger.warning(f"处理样本时出错: {str(e)}")
                idx = (idx + 1) % len(self.img_paths)
                continue
                
        raise RuntimeError(f"在 {self.max_retry} 次尝试后未能获取有效样本")
        
    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.img_paths)


# ---------------------- 训练模块 ----------------------
class DeepFashionTrainer:
    """DeepFashion多任务训练器"""
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 3e-4,
                 enable_landmarks: bool = True,
                 enable_segmentation: bool = True):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.enable_landmarks = enable_landmarks
        self.enable_segmentation = enable_segmentation
        
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
        self.landmark_criterion = nn.MSELoss()
        self.segmentation_criterion = nn.BCEWithLogitsLoss()
        
        # 多任务损失
        num_tasks = 1 + int(enable_landmarks) + int(enable_segmentation)
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
        
        # 获取其他任务的标签（如果有）
        landmark_labels = batch.get('landmarks')
        if landmark_labels is not None:
            landmark_labels = landmark_labels.to(self.device)
            
        seg_labels = batch.get('segmentation')
        if seg_labels is not None:
            seg_labels = seg_labels.to(self.device)
            
        # 清零梯度
        self.optimizer.zero_grad()
        
        # 使用AMP进行前向传播
        with torch.amp.autocast('cuda'):
            outputs = self.model(images)
            
            # 计算各任务的损失
            losses = []
            metrics = {}
            
            # 1. 属性分类损失
            attr_loss = self.attr_criterion(outputs['attr_logits'], attr_labels)
            losses.append(attr_loss)
            metrics['attr_loss'] = attr_loss.item()
            
            # 2. 关键点检测损失（如果启用）
            if self.enable_landmarks and landmark_labels is not None:
                landmark_coords = outputs['landmark_coords']
                landmark_vis = outputs['landmark_vis']
                # 只计算可见关键点的损失
                vis_mask = landmark_labels[:, :, 2] > 0
                if vis_mask.any():
                    landmark_loss = self.landmark_criterion(
                        landmark_coords[vis_mask],
                        landmark_labels[vis_mask, :2]
                    )
                    losses.append(landmark_loss)
                    metrics['landmark_loss'] = landmark_loss.item()
                
            # 3. 分割损失（如果启用）
            if self.enable_segmentation and seg_labels is not None:
                seg_loss = self.segmentation_criterion(
                    outputs['seg_logits'],
                    seg_labels
                )
                losses.append(seg_loss)
                metrics['seg_loss'] = seg_loss.item()
                
            # 计算总损失
            total_loss = self.multi_task_loss(torch.stack(losses))
            metrics['total_loss'] = total_loss.item()
            
        # 反向传播
        self.scaler.scale(total_loss).backward()
        
        # 梯度裁剪
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # 优化器步进
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 更新学习率
        self.scheduler.step()
        
        # 更新EMA模型
        if self.current_epoch >= self.ema_start:
            self.ema.update_parameters(self.model)
            
        return metrics

    def train_epoch(self):
        self.model.train()
        metrics = {
            'total_loss': 0.0,
            'attr_loss': 0.0,
            'landmark_loss': 0.0,
            'seg_loss': 0.0
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
            pbar.set_postfix({k: f'{v/len(pbar):.4f}' for k, v in metrics.items()})
            
        return {k: v/len(self.train_loader) for k, v in metrics.items()}
        
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
        landmark_error = 0.0
        landmark_count = 0
        seg_iou = 0.0
        seg_count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                attr_labels = batch['attr_labels'].to(self.device)
                
                # 获取其他任务的标签
                landmark_labels = batch.get('landmarks')
                if landmark_labels is not None:
                    landmark_labels = landmark_labels.to(self.device)
                    
                seg_labels = batch.get('segmentation')
                if seg_labels is not None:
                    seg_labels = seg_labels.to(self.device)
                    
                # 前向传播
                outputs = eval_model(images)
                
                # 1. 属性分类评估
                attr_preds = (outputs['attr_logits'] > 0.5).float()
                attr_correct += (attr_preds == attr_labels).float().sum().item()
                attr_total += attr_labels.numel()
                
                # 2. 关键点检测评估
                if self.enable_landmarks and landmark_labels is not None:
                    vis_mask = landmark_labels[:, :, 2] > 0
                    if vis_mask.any():
                        landmark_error += torch.norm(
                            outputs['landmark_coords'][vis_mask] - landmark_labels[vis_mask, :2],
                            dim=1
                        ).mean().item()
                        landmark_count += 1
                        
                # 3. 分割评估
                if self.enable_segmentation and seg_labels is not None:
                    seg_preds = (outputs['seg_logits'] > 0).float()
                    intersection = (seg_preds * seg_labels).sum().item()
                    union = (seg_preds + seg_labels).gt(0).sum().item()
                    if union > 0:
                        seg_iou += intersection / union
                        seg_count += 1
                        
        # 计算各项指标
        attr_acc = attr_correct / attr_total if attr_total > 0 else 0
        landmark_avg_error = landmark_error / landmark_count if landmark_count > 0 else 0
        seg_avg_iou = seg_iou / seg_count if seg_count > 0 else 0
        
        return {
            'attr_acc': attr_acc,
            'landmark_error': landmark_avg_error,
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
                # 保存最佳模型
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model.state_dict(), model_path)
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
    """处理批量数据的组合函数
    
    Args:
        batch: 一批数据样本的列表
        
    Returns:
        合并后的批量数据
    """
    # 过滤掉None（无效样本）
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}
        
    # 组合所有样本
    output = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
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
                sampler=train_sampler,
                shuffle=False
            )
            val_loader = create_safe_loader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                shuffle=False
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
                            best_fold_model,
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

            # 保存当前折的最佳模型
            torch.save(
                best_fold_model,
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


# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 1. 定义文件路径
    DEEPFASHION_ROOT = "deepfashion"
    CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
    ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")
    IMG_DIR = os.path.join(CATEGORY_ROOT, "Img", "img")  # 修正图像根目录路径
    
    # 训练集文件
    TRAIN_IMG_LIST = os.path.join(ANNO_DIR, "train.txt")
    TRAIN_ATTR_FILE = os.path.join(ANNO_DIR, "train_attr.txt")
    TRAIN_CATE_FILE = os.path.join(ANNO_DIR, "train_cate.txt")
    TRAIN_BBOX_FILE = os.path.join(ANNO_DIR, "train_bbox.txt")
    TRAIN_LANDMARK_FILE = os.path.join(ANNO_DIR, "train_landmark.txt")
    
    # 验证集文件
    VAL_IMG_LIST = os.path.join(ANNO_DIR, "val.txt")
    VAL_ATTR_FILE = os.path.join(ANNO_DIR, "val_attr.txt")
    VAL_CATE_FILE = os.path.join(ANNO_DIR, "val_cate.txt")
    VAL_BBOX_FILE = os.path.join(ANNO_DIR, "val_bbox.txt")
    VAL_LANDMARK_FILE = os.path.join(ANNO_DIR, "val_landmark.txt")
    
    # 2. 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 3. 创建数据集
    train_dataset = DeepFashionDataset(
        img_list_file=TRAIN_IMG_LIST,
        attr_file=TRAIN_ATTR_FILE,
        cate_file=TRAIN_CATE_FILE,
        bbox_file=TRAIN_BBOX_FILE,
        landmark_file=TRAIN_LANDMARK_FILE,
        image_dir=IMG_DIR,
        transform=train_transform
    )
    
    val_dataset = DeepFashionDataset(
        img_list_file=VAL_IMG_LIST,
        attr_file=VAL_ATTR_FILE,
        cate_file=VAL_CATE_FILE,
        bbox_file=VAL_BBOX_FILE,
        landmark_file=VAL_LANDMARK_FILE,
        image_dir=IMG_DIR,
        transform=val_transform
    )
    
    # 4. 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 5. 创建模型和训练器
    model = FullModel(
        num_classes=26,  # DeepFashion数据集的属性数量
        enable_landmarks=True
    )
    
    trainer = DeepFashionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        enable_landmarks=True
    )
    
    # 6. 开始训练
    trainer.train(epochs=50, save_dir="checkpoints")

    logger.info("训练完成！")
