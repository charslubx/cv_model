import torch
import logging
from training import MultiLabelDataset, get_transforms, create_safe_loader, Trainer
from base_model import FullModel
import pandas as pd
import kagglehub

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(train_csv, image_dir, epochs=10, num_classes=None):
    """训练模型"""
    try:
        # 创建数据转换
        train_transform = get_transforms(train=True)
        val_transform = get_transforms(train=False)

        # 创建数据集
        train_dataset = MultiLabelDataset(
            csv_path=train_csv,
            image_dir=image_dir,
            transform=train_transform  # 确保提供转换
        )
        
        logger.info(f"创建的训练数据集大小: {len(train_dataset)}")
        
        # 添加数据集有效性检查
        if len(train_dataset) < 10:
            raise ValueError(f"数据集样本不足 ({len(train_dataset)}), 至少需要10个有效样本")
            
        # 分割训练集和验证集 (80/20)
        total_size = len(train_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = create_safe_loader(
            train_subset,
            batch_size=32,
            shuffle=True
        )
        
        val_loader = create_safe_loader(
            val_subset,
            batch_size=32,
            shuffle=False
        )
        
        # 初始化模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        model = FullModel(num_classes=num_classes)
        trainer = Trainer(model, train_loader, val_loader, device)
        
        # 训练模型
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
    dataset_root = "/root/.cache/kagglehub/datasets/vishalbsadanand/deepfashion-1/versions/1/datasets"  # 根据实际路径调整
    
    # 准备数据集并获取属性数量
    num_attributes = prepare_deepfashion_data.prepare_deepfashion_dataset(
        dataset_root=dataset_root,
        min_samples_per_attr=100
    )
    
    # 训练模型
    # train_model(
    #     train_csv="data/train_labels.csv",
    #     image_dir="data/images",
    #     epochs=50,
    #     num_classes=num_attributes
    # )