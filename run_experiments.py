import torch
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

def create_models():
    """创建所有对比模型"""
    models = {}
    
    # 1. 基准模型（仅使用CNN特征）
    models['baseline'] = FullModel(
        num_classes=26,
        enable_segmentation=False,
        gat_dims=[2048],  # 最简单的线性变换
        gat_heads=1,
        cnn_type='resnet50',
        weights='IMAGENET1K_V1'
    )
    
    # 2. CNN+Attention模型
    models['cnn_attention'] = FullModel(
        num_classes=26,
        enable_segmentation=False,
        gat_dims=[2048, 1024],
        gat_heads=8,  # 使用多头注意力
        cnn_type='resnet50',
        weights='IMAGENET1K_V1'
    )
    
    # 3. 多尺度特征模型
    models['multi_scale'] = FullModel(
        num_classes=26,
        enable_segmentation=False,
        gat_dims=[2048, 1024],
        gat_heads=4,
        cnn_type='resnet50',
        weights='IMAGENET1K_V1'
    )
    
    # 4. GAT模型（完整版）
    models['gat'] = FullModel(
        num_classes=26,
        enable_segmentation=False,
        gat_dims=[2048, 1024, 512],
        gat_heads=4,
        cnn_type='resnet50',
        weights='IMAGENET1K_V1'
    )
    
    return models

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
        # 检查是否有训练历史数据
        if isinstance(model_results, dict):
            # 处理不同的数据结构
            if 'train' in model_results and 'val' in model_results:
                # 原始数据结构
                train_data = model_results['train']
                val_data = model_results['val']
            else:
                # 新的数据结构
                train_data = model_results.get('history', {}).get('train', [])
                val_data = model_results.get('history', {}).get('val', [])
            
            if train_data or val_data:
                plt.figure(figsize=(12, 6))
                
                if train_data:
                    train_df = pd.DataFrame(train_data)
                    if 'epoch' in train_df and 'loss' in train_df:
                        plt.plot(train_df['epoch'], train_df['loss'], label='Train Loss')
                
                if val_data:
                    val_df = pd.DataFrame(val_data)
                    if 'epoch' in val_df and 'loss' in val_df:
                        plt.plot(val_df['epoch'], val_df['loss'], label='Val Loss')
                
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
    
    # 创建模型
    logger.info("创建模型...")
    try:
        models = create_models()
        logger.info(f"创建了 {len(models)} 个模型用于比较")
    except Exception as e:
        logger.error(f"创建模型时出错: {str(e)}")
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