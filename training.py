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

from base_model import FullModel, FocalLoss

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
                width, height = img.size  # 使用size而不是shape
                if width < 10 or height < 10:
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
        self.label_columns = label_columns or [col for col in self.df.columns if col != 'image_path']
        self.max_retry = max_retry

        # 预加载所有图像尺寸信息
        self.image_sizes = {}
        self._preload_image_sizes()

        # 数据清洗步骤
        self._clean_data()

    def _preload_image_sizes(self):
        """预加载所有图像的尺寸信息"""
        logger.info("预加载图像尺寸信息...")
        valid_images = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="加载图像信息"):
            img_path = os.path.join(self.image_dir, row['image_path'])
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    if width >= 10 and height >= 10:  # 添加基本的尺寸验证
                        self.image_sizes[img_path] = (width, height)
                        valid_images.append(idx)
            except Exception as e:
                logger.warning(f"无法加载图像 {img_path}: {str(e)}")
                continue

        # 更新数据框，只保留有效图像
        self.df = self.df.loc[valid_images].reset_index(drop=True)
        logger.info(f"成功加载 {len(self.image_sizes)} 个有效图像")

    def __getitem__(self, idx):
        """获取数据集中的一个样本"""
        for _ in range(self.max_retry):
            try:
                row = self.df.iloc[idx]
                img_path = os.path.join(self.image_dir, row['image_path'])

                # 检查图像是否已预加载尺寸信息
                if img_path not in self.image_sizes:
                    idx = (idx + 1) % len(self.df)
                    continue

                # 加载图像
                try:
                    with Image.open(img_path) as img:
                        img = img.convert('RGB')
                        width, height = img.size  # 使用PIL的size属性

                        # 原始图像尺寸验证
                        if width < 10 or height < 10:
                            logger.warning(f"图像尺寸过小: {img_path}, size={img.size}")
                            idx = (idx + 1) % len(self.df)
                            continue

                        # 应用数据转换
                        if self.transform is not None:
                            try:
                                img_tensor = self.transform(img)  # 转换后得到张量
                                # 现在可以安全使用shape属性（张量维度为[C, H, W]）
                                if img_tensor.shape[1] < 10 or img_tensor.shape[2] < 10:
                                    raise ValueError(f"转换后尺寸过小: {img_tensor.shape}")
                                # print(img_tensor.shape)   # 应该类似torch.Size([3, 224, 224])
                            except Exception as e:
                                logger.warning(f"转换失败 {img_path}: {str(e)}")
                                idx = (idx + 1) % len(self.df)
                                continue
                        else:
                            # 默认转换使用size判断
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            img_tensor = transform(img)

                    # 转换标签为张量
                    labels = torch.FloatTensor(row[self.label_columns].values.astype(float))

                    return img_tensor, labels

                except Exception as e:
                    logger.warning(f"处理图像 {img_path} 时出错: {str(e)}")
                    idx = (idx + 1) % len(self.df)
                    continue

            except Exception as e:
                logger.warning(f"处理样本时出错: {str(e)}")
                idx = (idx + 1) % len(self.df)
                continue

        raise RuntimeError(f"在 {self.max_retry} 次尝试后未能获取有效样本")

    def _clean_data(self):
        """执行数据清洗"""
        original_size = len(self.df)
        logger.info(f"原始数据样本数: {original_size}")

        # 1. 只保留已成功预加载尺寸信息的图像
        valid_images = [idx for idx, row in self.df.iterrows()
                        if os.path.join(self.image_dir, row['image_path']) in self.image_sizes]
        self.df = self.df.loc[valid_images]
        logger.info(f"有效图像数: {len(self.df)}")

        # 2. 去除重复样本
        self.df = self.df.drop_duplicates(subset=['image_path'])
        logger.info(f"去重后样本数: {len(self.df)}")

        # 3. 确保标签列为数值类型
        for col in self.label_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 4. 去除包含NaN的行
        self.df = self.df.dropna(subset=self.label_columns)
        logger.info(f"去除NaN后样本数: {len(self.df)}")

        # 5. 确保所有标签都是0或1
        valid_labels = (self.df[self.label_columns] == 0) | (self.df[self.label_columns] == 1)
        self.df = self.df[valid_labels.all(axis=1)]
        logger.info(f"标签验证后样本数: {len(self.df)}")

        # 6. 确保每个样本至少有一个正标签
        has_positive = (self.df[self.label_columns].sum(axis=1) > 0)
        self.df = self.df[has_positive]
        logger.info(f"最终有效样本数: {len(self.df)}")

        if len(self.df) == 0:
            raise ValueError("清洗后没有剩余有效样本！")

        # 重置索引
        self.df = self.df.reset_index(drop=True)

        # 打印一些样本数据用于检查
        logger.info("\n数据样本示例:")
        logger.info(self.df[['image_path'] + self.label_columns].head())

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.df)


# ---------------------- 训练模块 ----------------------
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 3e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 调整优化器适应更大GAT
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.15  # 更强的权重衰减
        )

        # 修改为OneCycleLR调度器
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=50,  # 总训练轮数
            steps_per_epoch=len(train_loader),
            pct_start=0.3,  # 预热阶段占比
            div_factor=25.0,  # 初始学习率 = max_lr/25
            final_div_factor=1e4  # 最终学习率 = max_lr/10000
        )

        # 3. 添加指数移动平均
        self.ema = torch.optim.swa_utils.AveragedModel(model)
        self.ema_start = 20  # 从第20轮开始使用EMA

        # 4. 调整FocalLoss参数
        self.criterion = FocalLoss(gamma=4.0, alpha=0.5)

        # 5. 添加梯度缩放器
        self.scaler = torch.cuda.amp.GradScaler()

        # 6. 调整早停参数
        self.patience = 15  # 增加耐心值
        self.best_val_acc = 0
        self.patience_counter = 0

        # 添加当前epoch计数器
        self.current_epoch = 0  # 新增初始化

        # 多GPU支持
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {torch.cuda.device_count()} GPUs!")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(self.train_loader, desc='Training'):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # 1. 添加标签平滑
            labels = labels * 0.9 + 0.05

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(self.device == "cuda")):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            # 2. 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.gat.parameters(),  # 仅裁剪GAT部分
                max_norm=1.0
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)

            # 每个batch后更新学习率
            self.scheduler.step()  # OneCycleLR在每个batch后调用

        # 更新EMA模型
        if self.current_epoch >= self.ema_start:
            self.ema.update_parameters(self.model)

        return total_loss / len(self.train_loader.dataset)

    def validate(self):
        # 使用EMA模型进行验证
        if self.current_epoch >= self.ema_start:
            eval_model = self.ema.module
        else:
            eval_model = self.model

        eval_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = eval_model(images)
                loss = self.criterion(outputs, labels)

                preds = (outputs > 0.5).float()
                correct += (preds == labels).float().sum().item()
                total += labels.numel()

                total_loss += loss.item() * images.size(0)

        return total_loss / len(self.val_loader.dataset), correct / total

    def train(self, epochs: int = 50):
        # 更新scheduler的epochs参数
        self.scheduler.total_steps = epochs * len(self.train_loader)

        for epoch in range(epochs):
            self.current_epoch = epoch + 1  # 更新当前epoch计数（从1开始）
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 添加早停机制
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break


# ---------------------- 改进的数据增强配置 ----------------------
def get_transforms(train=True):
    """获取增强的数据转换"""
    base_transform = [
        # 所有PIL图像操作在前
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
    ]

    if train:
        return transforms.Compose([
            *base_transform,
            # 张量操作在后
            transforms.ToTensor(),  # 转换PIL图像为张量
            transforms.RandomErasing(p=0.2),  # 需要张量输入
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # 先转换为张量
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
            trainer = Trainer(model, train_loader, val_loader)

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

                fold_result['train_loss'].append(train_loss)
                fold_result['val_loss'].append(val_loss)
                fold_result['val_acc'].append(val_acc)

                # 保存最佳模型
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    best_fold_model = copy.deepcopy(model.state_dict())

                    # 如果是总体最佳，则额外保存
                    if val_acc > overall_best_acc:
                        overall_best_acc = val_acc
                        torch.save(
                            best_fold_model,
                            os.path.join(self.save_dir, f'best_model_overall.pth')
                        )

                logger.info(
                    f"Fold {fold + 1} Epoch {epoch + 1}/{self.epochs_per_fold} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Best Acc: {best_fold_acc:.4f}"
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

    # 初始化交叉验证训练器
    kfold_trainer = KFoldTrainer(
        dataset=train_dataset,
        num_folds=5,
        batch_size=32,
        epochs_per_fold=40
    )

    # 执行交叉验证
    kfold_trainer.run()

    # 获取集成模型
    ensemble_model = kfold_trainer.get_ensemble_model()
