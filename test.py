import torch
from PIL import Image
import logging
from base_model import FullModel, ImageGraphPipeline
from training import MultiLabelDataset, get_transforms, create_safe_loader, Trainer
import os
import pandas as pd

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义数据集路径
DEEPFASHION_ROOT = "deepfashion/Category and Attribute Prediction Benchmark"
IMG_DIR = os.path.join(DEEPFASHION_ROOT, "Img/img")  # 使用解压后的img目录
ANNO_DIR = os.path.join(DEEPFASHION_ROOT, "Anno_fine")  # 使用细粒度标注
ATTR_CLOTH_FILE = os.path.join(ANNO_DIR, "list_attr_cloth.txt")  # 属性定义文件

def test_single_image(model_path, image_path):
    """测试单张图片"""
    try:
        # 加载模型
        model = FullModel(num_classes=20)
        model.load_state_dict(torch.load(model_path))
        pipeline = ImageGraphPipeline(model)
        
        # 加载属性名称
        with open(ATTR_CLOTH_FILE, 'r') as f:
            f.readline()  # 跳过第一行（属性数量）
            f.readline()  # 跳过第二行（表头）
            attr_names = [line.strip().split()[0] for line in f]
        
        # 预测
        output = pipeline(image_path)
        predictions = (torch.sigmoid(output) > 0.5).float()
        
        # 输出预测结果
        logger.info("预测结果:")
        for i, (pred, attr_name) in enumerate(zip(predictions, attr_names)):
            if pred == 1:
                logger.info(f"- {attr_name}")
        
        return predictions
        
    except Exception as e:
        logger.error(f"单张图片测试失败: {str(e)}")
        raise

def test_batch_images(model_path, test_attr_file):
    """测试批量图片"""
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not os.path.exists(test_attr_file):
            raise FileNotFoundError(f"测试标签文件不存在: {test_attr_file}")
            
        # 加载数据
        test_dataset = MultiLabelDataset(
            attr_file=test_attr_file,
            transform=get_transforms(train=False)
        )
        test_loader = create_safe_loader(test_dataset, batch_size=32)
        
        # 加载模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FullModel(num_classes=20)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()
        
        # 加载属性名称
        with open(ATTR_CLOTH_FILE, 'r') as f:
            f.readline()  # 跳过第一行（属性数量）
            f.readline()  # 跳过第二行（表头）
            attr_names = [line.strip().split()[0] for line in f]
        
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
        for i, (acc, attr_name) in enumerate(zip(per_class_acc, attr_names)):
            logger.info(f"属性 {attr_name} 准确率: {acc:.4f}")
        
        accuracy = correct / total
        logger.info(f"测试集准确率: {accuracy:.4f}")
        
        # 添加内存使用情况日志
        if torch.cuda.is_available():
            logger.info(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
            
        return accuracy
        
    except Exception as e:
        logger.error(f"批量测试失败: {str(e)}")
        raise

if __name__ == "__main__":
    # 测试单张图片
    try:
        single_pred = test_single_image(
            model_path="multilabel_model.pth",
            image_path=os.path.join(IMG_DIR, "test_image.jpg")
        )
        logger.info("单张图片测试完成")
    except Exception as e:
        logger.error(f"单张图片测试失败: {str(e)}")
    
    # 测试批量图片
    try:
        batch_acc = test_batch_images(
            model_path="multilabel_model.pth",
            test_attr_file=os.path.join(ANNO_DIR, "test_attr.txt")
        )
        logger.info("批量测试完成")
    except Exception as e:
        logger.error(f"批量测试失败: {str(e)}")
