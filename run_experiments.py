import torch
import torch.nn as nn
import torchvision.models as models
import logging
from experiment_config import EXPERIMENT_CONFIG, validate_dataset_structure
from experiment_runner import ExperimentRunner
from base_model import FullModel
from dataset import create_data_loaders
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import sys
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V1')
        self.model.fc = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        return {'attr_logits': self.model(x)}

class MultiLabelEfficientNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        state_dict = torch.load('checkpoints/efficientnet_b0_rwightman-3dd342df.pth')
        self.model.load_state_dict(state_dict)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return {'attr_logits': self.model(x)}

class MultiLabelDenseNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = models.densenet121(weights='IMAGENET1K_V1')
        self.model.classifier = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        return {'attr_logits': self.model(x)}

class MultiLabelSwinTransformer(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.model = models.swin_t(weights=None)
        state_dict = torch.load('checkpoints/swin_t-704ceda3.pth')
        self.model.load_state_dict(state_dict)
        self.model.head = nn.Linear(768, num_classes)
    
    def forward(self, x):
        return {'attr_logits': self.model(x)}

def load_pretrained_models():
    """加载预训练模型"""
    model_dict = {}
    
    # 1. 加载我们的最佳模型
    our_model = FullModel(
        num_classes=26,
        enable_segmentation=False,
        gat_dims=[2048, 1024],
        gat_heads=8,
        cnn_type='resnet50',
        weights='IMAGENET1K_V1'
    )
    our_model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    model_dict['our_model'] = our_model
    
    # 4. 加载预训练的ResNet50模型
    resnet_model = MultiLabelResNet(num_classes=26)
    model_dict['resnet50'] = resnet_model
    
    # 5. 加载预训练的EfficientNet模型
    efficient_model = MultiLabelEfficientNet(num_classes=26)
    model_dict['efficientnet'] = efficient_model
    
    # 6. 加载预训练的DenseNet模型
    densenet_model = MultiLabelDenseNet(num_classes=26)
    model_dict['densenet'] = densenet_model
    
    # 7. 加载预训练的Swin Transformer模型
    swin_model = MultiLabelSwinTransformer(num_classes=26)
    model_dict['swin_transformer'] = swin_model
    
    return model_dict

def plot_results(results, save_dir='results'):
    """绘制实验结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查结果数据结构
    if not results:
        logger.warning("没有结果数据可供绘制")
        return
        
    # 1. 模型对比图（如果有F1分数）
    if isinstance(results, dict) and any('best_f1' in metrics for metrics in results.values()):
        model_comparison = pd.DataFrame(
            [(model, metrics.get('best_f1', 0)) for model, metrics in results.items()],
            columns=['Model', 'F1 Score']
        )
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=model_comparison, x='Model', y='F1 Score')
        plt.title('Model Comparison - F1 Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'model_comparison.png'))
        plt.close()
    
    # 2. 训练曲线（如果有训练历史）
    for model_name, model_results in results.items():
        if isinstance(model_results, dict) and 'history' in model_results:
            history = model_results['history']
            if 'train' in history and 'val' in history:
                plt.figure(figsize=(12, 6))
                
                train_data = pd.DataFrame(history['train'])
                val_data = pd.DataFrame(history['val'])
                
                if 'epoch' in train_data and 'loss' in train_data:
                    plt.plot(train_data['epoch'], train_data['loss'], label='Train Loss')
                if 'epoch' in val_data and 'loss' in val_data:
                    plt.plot(val_data['epoch'], val_data['loss'], label='Val Loss')
                
                plt.title(f'{model_name} Training Progress')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{model_name}_training.png'))
                plt.close()
        
        # 保存原始结果数据
        with open(os.path.join(save_dir, f'{model_name}_results.json'), 'w') as f:
            json.dump(model_results, f, indent=4)

def main():
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 验证数据集结构
    logger.info("验证数据集结构...")
    if not validate_dataset_structure():
        logger.error("数据集结构验证失败，请检查数据集路径和文件结构")
        sys.exit(1)
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    try:
        train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
        logger.info(f"训练集大小: {len(train_loader.dataset)}")
        logger.info(f"验证集大小: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"创建数据加载器时出错: {str(e)}")
        sys.exit(1)
    
    # 加载预训练模型
    logger.info("加载预训练模型...")
    try:
        models = load_pretrained_models()
        logger.info(f"加载了 {len(models)} 个预训练模型用于比较")
    except Exception as e:
        logger.error(f"加载预训练模型时出错: {str(e)}")
        sys.exit(1)
    
    # 创建实验运行器
    runner = ExperimentRunner(
        config=EXPERIMENT_CONFIG,
        models=models,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # 运行主要实验
    logger.info("开始运行主要实验...")
    results = runner.run_experiment()
    
    # 绘制结果
    logger.info("绘制实验结果...")
    plot_results(results, save_dir='results')
    
    logger.info("实验完成!")

if __name__ == '__main__':
    main() 