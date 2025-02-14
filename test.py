import torch
from PIL import Image
import logging
from base_model import FullModel, ImageGraphPipeline
from training import MultiLabelDataset, get_transforms, create_safe_loader, Trainer
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_image(model_path, image_path):
    """测试单张图片"""
    # 加载模型
    model = FullModel(num_classes=20)
    model.load_state_dict(torch.load(model_path))
    pipeline = ImageGraphPipeline(model)
    
    # 预测
    output = pipeline(image_path)
    predictions = (torch.sigmoid(output) > 0.5).float()
    
    logger.info(f"预测结果: {predictions}")
    return predictions

def test_batch_images(model_path, test_csv, image_dir):
    """测试批量图片"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(test_csv):
            raise FileNotFoundError(f"测试标签文件不存在: {test_csv}")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图片目录不存在: {image_dir}")
            
        # 加载数据
        test_dataset = MultiLabelDataset(
            csv_path=test_csv,
            image_dir=image_dir,
            transform=get_transforms(train=False)
        )
        test_loader = create_safe_loader(test_dataset, batch_size=32)
        
        # 加载模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FullModel(num_classes=20)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        
        # 评估
        correct = 0
        total = 0
        
        # 添加更多评估指标
        predictions_all = []
        labels_all = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                
                predictions_all.append(predictions.cpu())
                labels_all.append(labels.cpu())
                
                correct += (predictions == labels).all(dim=1).sum().item()
                total += images.size(0)
        
        # 合并所有预测和标签
        predictions_all = torch.cat(predictions_all)
        labels_all = torch.cat(labels_all)
        
        # 计算每个类别的准确率
        per_class_acc = (predictions_all == labels_all).float().mean(0)
        for i, acc in enumerate(per_class_acc):
            logger.info(f"类别 {i} 准确率: {acc:.4f}")
        
        accuracy = correct / total
        logger.info(f"测试集准确率: {accuracy:.4f}")
        
        # 添加内存使用情况日志
        if torch.cuda.is_available():
            logger.info(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
        return accuracy
        
    except Exception as e:
        logger.error(f"测试过程发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试单张图片
    try:
        single_pred = test_single_image(
            model_path="multilabel_model.pth",
            image_path="data/images/test_image.jpg"
        )
        logger.info("单张图片测试完成")
    except Exception as e:
        logger.error(f"单张图片测试失败: {str(e)}")
    
    # 测试批量图片
    try:
        batch_acc = test_batch_images(
            model_path="multilabel_model.pth",
            test_csv="data/test_labels.csv",
            image_dir="data/images"
        )
        logger.info("批量测试完成")
    except Exception as e:
        logger.error(f"批量测试失败: {str(e)}")
