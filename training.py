import json
import logging
import os

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

from base_model import FullModel

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置常量加载
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 允许加载截断的图像


# ---------------------- 增强的数据验证类 ----------------------
class DataValidator:
    @staticmethod
    def validate_image(img_path):
        """验证图像文件"""
        if not os.path.exists(img_path):
            logger.warning(f"图像文件不存在: {img_path}")
            return False
        
        try:
            with Image.open(img_path) as img:
                # 检查图像是否可以正常打开
                img.verify()
                # 重新打开图像以检查尺寸
                img = Image.open(img_path)
                if img.size[0] < 10 or img.size[1] < 10:
                    logger.warning(f"图像尺寸过小: {img_path}, size={img.size}")
                    return False
                return True
        except Exception as e:
            logger.warning(f"图像验证失败: {img_path}, 错误: {str(e)}")
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
class MultiLabelDataset(Dataset):
    def __init__(self,
                 csv_path: str,
                 image_dir: str,
                 transform=None,
                 label_columns: list = None,
                 max_retry: int = 5):
        """
        初始化数据集
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform
        # 如果未指定标签列，则假设除了image_path之外的所有列都是标签列
        self.label_columns = label_columns or [col for col in self.df.columns if col != 'image_path']
        self.max_retry = max_retry

        # 数据清洗步骤
        self._clean_data()

    def _clean_data(self):
        """执行数据清洗"""
        original_size = len(self.df)
        logger.info(f"原始数据样本数: {original_size}")

        # 1. 去除重复样本
        self.df = self.df.drop_duplicates(subset=['image_path'])
        logger.info(f"去重后样本数: {len(self.df)}")

        # 2. 确保标签列为数值类型并打印数据类型信息
        logger.info("标签列数据类型转换前:")
        logger.info(self.df[self.label_columns].dtypes)
        
        # 只对标签列进行数值转换
        label_df = self.df[self.label_columns].copy()
        for col in self.label_columns:
            try:
                label_df[col] = pd.to_numeric(label_df[col], errors='coerce')
                invalid_count = label_df[col].isna().sum()
                if invalid_count > 0:
                    logger.warning(f"列 {col} 中有 {invalid_count} 个无效值被转换为 NaN")
            except Exception as e:
                logger.error(f"列 {col} 转换为数值类型时出错: {str(e)}")
                problem_samples = self.df[pd.to_numeric(self.df[col], errors='coerce').isna()]
                logger.error(f"问题数据样本:\n{problem_samples[col].head()}")
                raise

        # 更新原始DataFrame中的标签列
        self.df[self.label_columns] = label_df

        logger.info("标签列数据类型转换后:")
        logger.info(self.df[self.label_columns].dtypes)
        logger.info(f"数值转换后样本数: {len(self.df)}")

        # 3. 去除包含 NaN 的行（只检查标签列）
        before_dropna = len(self.df)
        self.df = self.df.dropna(subset=self.label_columns)
        logger.info(f"去除NaN后样本数: {len(self.df)} (删除了 {before_dropna - len(self.df)} 行)")

        # 4. 去除无效标签（确保所有值都是 0 或 1）
        before_valid = len(self.df)
        valid_labels = (self.df[self.label_columns] == 0) | (self.df[self.label_columns] == 1)
        self.df = self.df[valid_labels.all(axis=1)]
        logger.info(f"去除非0/1值后样本数: {len(self.df)} (删除了 {before_valid - len(self.df)} 行)")

        # 打印一些样本数据用于检查
        if len(self.df) > 0:
            logger.info("数据样本示例:")
            logger.info(self.df[['image_path'] + self.label_columns].head())
        
        # 5. 去除没有正样本的行
        before_positive = len(self.df)
        row_sums = self.df[self.label_columns].sum(axis=1)
        self.df = self.df[row_sums > 0]
        logger.info(f"去除无正样本后样本数: {len(self.df)} (删除了 {before_positive - len(self.df)} 行)")

        # 6. 记录清洗结果
        cleaned_size = len(self.df)
        logger.info(f"数据清洗最终结果: {original_size} -> {cleaned_size} 个样本")
        
        if cleaned_size == 0:
            if before_positive > 0:
                logger.error("最后一步前的数据示例:")
                logger.error(self.df[['image_path'] + self.label_columns].head())
                logger.error(f"行和统计: {row_sums.describe()}")
            raise ValueError("清洗后没有剩余有效样本！请检查数据格式是否正确。")

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        for _ in range(self.max_retry):
            try:
                row = self.df.iloc[idx]
                img_path = os.path.join(self.image_dir, row['image_path'])

                # 验证图像文件
                if not DataValidator.validate_image(img_path):
                    idx = (idx + 1) % len(self.df)
                    continue

                # 验证标签
                labels = row[self.label_columns]
                if not DataValidator.validate_labels(labels):
                    idx = (idx + 1) % len(self.df)
                    continue

                # 加载并转换图像
                image = Image.open(img_path).convert('RGB')
                
                # 应用数据转换
                if self.transform is not None:
                    image = self.transform(image)
                else:
                    # 默认转换
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        )
                    ])
                    image = transform(image)

                # 转换标签为张量
                labels = torch.FloatTensor(labels.values.astype(float))
                
                return image, labels

            except Exception as e:
                logger.error(f"处理图像 {img_path} 时出错: {str(e)}")
                idx = (idx + 1) % len(self.df)
                continue

        raise RuntimeError(f"在 {self.max_retry} 次尝试后未能获取有效样本")

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.df)


# ---------------------- 训练模块 ----------------------
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 添加混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")

        # 多标签分类使用BCEWithLogitsLoss
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.1)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)  # 异步传输
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # 混合精度训练
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)

        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)
                # 计算准确率（阈值0.5）
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).all(dim=1).sum().item()
                total += images.size(0)

        return total_loss / len(self.val_loader.dataset), correct / total

    def train(self, epochs: int = 10):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()

            print(f"Epoch {epoch + 1}/{epochs}")
            print(
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")


# ---------------------- 数据增强配置 ----------------------
def get_transforms(train=True):
    """获取数据转换"""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ---------------------- 改进的DataLoader配置 ----------------------
def create_safe_loader(dataset: Dataset,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       sampler=None,
                       shuffle: bool = None) -> DataLoader:
    """
    创建带有异常处理的DataLoader
    sampler: 自定义采样器
    shuffle: 显式控制是否打乱（当使用sampler时自动禁用）
    """

    def collate_fn(batch):
        # 过滤掉None（无效样本）
        batch = [b for b in batch if b is not None]
        return torch.utils.data.dataloader.default_collate(batch)

    # 自动处理shuffle逻辑
    if sampler is not None:
        shuffle = False  # 使用sampler时必须禁用shuffle
    elif shuffle is None:
        shuffle = True  # 默认行为

    # 优化数据加载配置
    num_workers = min(os.cpu_count(), 8)  # 自动设置合理的工作进程数
    pin_memory = pin_memory and torch.cuda.is_available()  # 自动启用pin_memory
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,  # 加速GPU数据传输
        persistent_workers=True if num_workers > 0 else False,  # 保持工作进程
        collate_fn=collate_fn,
        worker_init_fn=lambda worker_id: np.random.seed(torch.initial_seed() % 2 ** 32),
        drop_last=True  # 避免最后批次尺寸不一致
    )


class KFoldTrainer:
    """K折交叉验证训练器"""

    def __init__(self,
                 dataset: Dataset,
                 num_folds: int = 5,
                 batch_size: int = 32,
                 epochs_per_fold: int = 10):
        self.dataset = dataset
        self.num_folds = num_folds
        self.batch_size = batch_size
        self.epochs_per_fold = epochs_per_fold
        self.kfold = KFold(n_splits=num_folds, shuffle=True)

        # 存储各折结果
        self.fold_results = []

    def run(self):
        """执行交叉验证"""
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(self.dataset)):
            logger.info(f"\n{'=' * 30} Fold {fold + 1}/{self.num_folds} {'=' * 30}")

            # 创建采样器
            train_sampler = SubsetRandomSampler(train_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            # 创建安全的数据加载器（显式设置shuffle=False）
            train_loader = create_safe_loader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                shuffle=False  # 采样器已处理顺序
            )
            val_loader = create_safe_loader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=val_sampler,
                shuffle=False
            )

            # 初始化全新模型
            model = FullModel(num_classes=20)
            trainer = Trainer(model, train_loader, val_loader)

            # 训练当前折
            fold_result = {
                'train_loss': [],
                'val_loss': [],
                'val_acc': []
            }
            for epoch in range(self.epochs_per_fold):
                train_loss = trainer.train_epoch()
                val_loss, val_acc = trainer.validate()
                trainer.scheduler.step()

                fold_result['train_loss'].append(train_loss)
                fold_result['val_loss'].append(val_loss)
                fold_result['val_acc'].append(val_acc)

                logger.info(
                    f"Fold {fold + 1} Epoch {epoch + 1}/{self.epochs_per_fold} | "
                    f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}"
                )

            self.fold_results.append(fold_result)

        self._analyze_results()

    def _analyze_results(self):
        """分析交叉验证结果"""
        best_acc = max([max(fold['val_acc']) for fold in self.fold_results])
        avg_acc = sum([fold['val_acc'][-1] for fold in self.fold_results]) / self.num_folds

        logger.info("\n=== Cross Validation Summary ===")
        logger.info(f"Best Val Acc Across Folds: {best_acc:.4f}")
        logger.info(f"Average Final Val Acc: {avg_acc:.4f}")

        # 可视化训练曲线
        self._plot_learning_curves()

    def _plot_learning_curves(self):
        """绘制学习曲线"""

        plt.figure(figsize=(12, 5))

        # 训练损失
        plt.subplot(1, 2, 1)
        for i, fold in enumerate(self.fold_results):
            plt.plot(fold['train_loss'], label=f'Fold {i + 1}')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # 验证准确率
        plt.subplot(1, 2, 2)
        for i, fold in enumerate(self.fold_results):
            plt.plot(fold['val_acc'], label=f'Fold {i + 1}')
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')

        plt.legend()
        plt.tight_layout()
        plt.savefig('cross_val_results.png')
        plt.close()


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
        original_trainer_init = Trainer.__init__

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
            Trainer.__init__ = patched_trainer_init
            kfold_trainer.run()
            avg_val_loss = sum(
                [fold['val_loss'][-1] for fold in kfold_trainer.fold_results]) / self.k_folds
        finally:
            Trainer.__init__ = original_trainer_init  # 恢复原始初始化

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
    # 1. 准备数据
    train_dataset = MultiLabelDataset(
        csv_path="data/train_labels.csv",
        image_dir="data/images",
        transform=get_transforms(train=True),
        label_columns=[f"class_{i}" for i in range(20)]
    )

    val_dataset = MultiLabelDataset(
        csv_path="data/val_labels.csv",
        image_dir="data/images",
        transform=get_transforms(train=False),
        label_columns=[f"class_{i}" for i in range(20)]
    )

    train_loader = create_safe_loader(train_dataset, batch_size=32)
    val_loader = create_safe_loader(val_dataset, batch_size=32)

    # 2. 初始化模型
    model = FullModel(num_classes=20)

    # 3. 开始训练
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(epochs=20)

    # 4. 保存模型
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), "multilabel_model.pth")
    else:
        torch.save(model.state_dict(), "multilabel_model.pth")
