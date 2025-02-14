import torch
import logging
from training import MultiLabelDataset, get_transforms, create_safe_loader, Trainer
from base_model import FullModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(train_csv, image_dir, epochs=10):
    """训练模型"""
    try:
        # 创建数据集
        train_dataset = MultiLabelDataset(
            csv_path=train_csv,
            image_dir=image_dir,
            transform=get_transforms(train=True)
        )
        
        # 分割训练集和验证集 (80/20)
        total_size = len(train_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # 创建数据加载器
        train_loader = create_safe_loader(train_dataset, batch_size=32)
        val_loader = create_safe_loader(val_dataset, batch_size=32)
        
        # 初始化模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {device}")
        
        model = FullModel(num_classes=20)
        trainer = Trainer(model, train_loader, val_loader)
        
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
    # 准备CIFAR-10数据
    import prepare_cifar_data
    prepare_cifar_data.prepare_cifar10_dataset()
    
    # 训练模型
    # train_model(
    #     train_csv="data/train_labels.csv",
    #     image_dir="data/images",
    #     epochs=50  # 增加到50轮
    # )