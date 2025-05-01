import torch
import logging
from training import MultiLabelDataset, get_transforms, create_safe_loader, Trainer
from base_model import FullModel
import pandas as pd
import kagglehub

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(train_csv, val_csv, image_dir, epochs=50, num_classes=None):
    """训练模型"""
    try:
        # 创建数据转换
        train_transform = get_transforms(train=True)
        val_transform = get_transforms(train=False)

        # 创建数据集
        train_dataset = MultiLabelDataset(
            csv_path=train_csv,
            image_dir=image_dir,
            transform=train_transform
        )

        val_dataset = MultiLabelDataset(
            csv_path=val_csv,
            image_dir=image_dir,
            transform=val_transform
        )

        logger.info(f"创建的训练数据集大小: {len(train_dataset)}")
        logger.info(f"创建的验证数据集大小: {len(val_dataset)}")

        # 添加数据集有效性检查
        if len(train_dataset) < 10:
            raise ValueError(f"训练集样本不足 ({len(train_dataset)}), 至少需要10个有效样本")
        if len(val_dataset) < 5:
            raise ValueError(f"验证集样本不足 ({len(val_dataset)}), 至少需要5个有效样本")

        # 创建数据加载器
        train_loader = create_safe_loader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )

        val_loader = create_safe_loader(
            val_dataset,
            batch_size=32,
            shuffle=False
        )

        # 初始化模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")

        model = FullModel(num_classes=num_classes)
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=3e-4
        )

        # 开始训练
        logger.info("开始训练...")
        trainer.train(epochs=epochs)

        # 保存模型
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), "multilabel_model.pth")
        else:
            torch.save(model.state_dict(), "multilabel_model.pth")

        logger.info("模型训练完成并保存!")

    except Exception as e:
        logger.error(f"训练过程发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    # 准备DeepFashion数据
    import prepare_deepfashion_data

    # 设置数据集路径
    dataset_root = "/databases/archive/datasets"

    # 准备数据集并获取属性数量
    num_attributes = prepare_deepfashion_data.prepare_deepfashion_dataset(
        dataset_root=dataset_root,
        min_samples_per_attr=100
    )

    # 训练模型
    train_model(
        train_csv="data/train_labels.csv",
        val_csv="data/val_labels.csv",
        image_dir="data/images",
        epochs=50,
        num_classes=num_attributes
    )