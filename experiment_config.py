import torch
import torch.nn as nn
import os
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 获取当前文件的绝对路径
CURRENT_FILE = os.path.abspath(__file__)
# 获取当前文件所在目录
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
# 获取工作目录（当前文件所在目录）
WORKSPACE_ROOT = CURRENT_DIR
logger.info(f"当前文件: {CURRENT_FILE}")
logger.info(f"当前目录: {CURRENT_DIR}")
logger.info(f"工作目录: {WORKSPACE_ROOT}")

# 数据集根目录
DEEPFASHION_ROOT = "/home/cv_model/DeepFashion"
CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
CATEGORY_IMG_DIR = os.path.join(CATEGORY_ROOT, "Img")
CATEGORY_ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")

# 训练集文件
TRAIN_IMG_LIST = os.path.join(CATEGORY_ANNO_DIR, "train.txt")
TRAIN_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_attr.txt")
TRAIN_CATE_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_cate.txt")
TRAIN_BBOX_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_bbox.txt")
TRAIN_SEG_FILE = os.path.join(CATEGORY_ANNO_DIR, "train_seg.txt")

# 验证集文件
VAL_IMG_LIST = os.path.join(CATEGORY_ANNO_DIR, "val.txt")
VAL_ATTR_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_attr.txt")
VAL_CATE_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_cate.txt")
VAL_BBOX_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_bbox.txt")
VAL_SEG_FILE = os.path.join(CATEGORY_ANNO_DIR, "val_seg.txt")

# 实验配置
EXPERIMENT_CONFIG = {
    'data': {
        'root_dir': CATEGORY_IMG_DIR,
        'train_img_list_file': TRAIN_IMG_LIST,
        'train_attr_file': TRAIN_ATTR_FILE,
        'train_cate_file': TRAIN_CATE_FILE,
        'train_bbox_file': TRAIN_BBOX_FILE,
        'train_seg_file': TRAIN_SEG_FILE,
        'val_img_list_file': VAL_IMG_LIST,
        'val_attr_file': VAL_ATTR_FILE,
        'val_cate_file': VAL_CATE_FILE,
        'val_bbox_file': VAL_BBOX_FILE,
        'val_seg_file': VAL_SEG_FILE,
        'batch_size': 32,
        'num_workers': 4,
        'max_retry': 5
    },
    'training': {
        'epochs': 50,
        'learning_rate': 3e-4,
        'weight_decay': 0.15,
        'early_stopping_patience': 15,
        'ema_start': 20,
        'scheduler_patience': 5
    },
    'model': {
        'num_classes': 26,
        'cnn_type': 'resnet50',
        'weights': 'IMAGENET1K_V1',
        'gat_dims': [2048, 1024, 512],
        'gat_heads': 4,
        'gat_dropout': 0.2
    }
}

def validate_dataset_structure():
    """验证数据集结构和文件是否存在"""
    required_files = [
        TRAIN_IMG_LIST,
        TRAIN_ATTR_FILE,
        VAL_IMG_LIST,
        VAL_ATTR_FILE
    ]
    
    # 检查必要的目录
    if not os.path.exists(CATEGORY_ROOT):
        logger.error(f"找不到数据集根目录: {CATEGORY_ROOT}")
        return False
        
    if not os.path.exists(CATEGORY_IMG_DIR):
        logger.error(f"找不到图像目录: {CATEGORY_IMG_DIR}")
        return False
        
    if not os.path.exists(CATEGORY_ANNO_DIR):
        logger.error(f"找不到标注目录: {CATEGORY_ANNO_DIR}")
        return False
    
    # 检查必要的文件
    for file_path in required_files:
        if not os.path.exists(file_path):
            logger.error(f"找不到必要的文件: {file_path}")
            return False
        else:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"文件为空: {file_path}")
                return False
            logger.info(f"文件 {Path(file_path).name} 存在，大小: {file_size/1024:.2f}KB")
    
    return True

# 评估指标类
class MetricsCalculator:
    @staticmethod
    def calculate_metrics(predictions, targets, threshold=0.5):
        """计算多个评估指标"""
        binary_preds = (predictions > threshold).float()
        
        # 计算每个属性的指标
        tp = (binary_preds * targets).sum(dim=0)
        fp = (binary_preds * (1 - targets)).sum(dim=0)
        fn = ((1 - binary_preds) * targets).sum(dim=0)
        
        # 计算精确率和召回率
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item()
        }

# 实验结果记录器
class ExperimentLogger:
    def __init__(self, experiment_name, models):
        self.experiment_name = experiment_name
        self.models = models
        self.results = {model: {'train': [], 'val': []} for model in models}
    
    def log_metrics(self, model_name, phase, epoch, metrics):
        """记录训练或验证指标"""
        self.results[model_name][phase].append({
            'epoch': epoch,
            **metrics
        })
    
    def save_results(self, save_path):
        """保存实验结果"""
        import json
        import os
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        result_file = os.path.join(save_path, f'{self.experiment_name}_results.json')
        with open(result_file, 'w') as f:
            json.dump(self.results, f, indent=4)

# 验证数据集结构
if not validate_dataset_structure():
    logger.error("数据集结构验证失败，请检查数据集路径和文件结构") 