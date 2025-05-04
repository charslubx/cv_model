import os
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from typing import Dict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器，提供全面的评估指标和可视化"""
    
    def __init__(self, model, data_loader, device, attr_names):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            data_loader: 数据加载器
            device: 计算设备
            attr_names: 属性名称列表
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.attr_names = attr_names
        
    def evaluate(self, save_dir: str = "evaluation_results") -> Dict:
        """
        执行全面评估
        
        Args:
            save_dir: 结果保存目录
            
        Returns:
            dict: 包含所有评估指标的字典
        """
        os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        
        # 收集预测和标签
        all_preds = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for batch in self.data_loader:
                images = batch['image'].to(self.device)
                labels = batch['attr_labels'].to(self.device)
                
                # 测量推理时间
                start_time = time.time()
                outputs = self.model(images)
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # 获取预测结果
                preds = torch.sigmoid(outputs['attr_logits'])
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # 合并所有预测和标签
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 计算平均推理时间
        avg_inference_time = np.mean(inference_times)
        
        # 计算总体指标
        metrics = self._compute_overall_metrics(all_preds, all_labels)
        metrics['inference_time'] = avg_inference_time
        
        # 计算每个属性的指标
        attr_metrics = self._compute_per_attribute_metrics(all_preds, all_labels)
        metrics['per_attribute'] = attr_metrics
        
        # 生成可视化
        self._generate_visualizations(all_preds, all_labels, save_dir)
        
        return metrics
    
    def _compute_overall_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict:
        """计算总体评估指标"""
        # 使用0.5作为阈值
        binary_preds = (preds > 0.5).float()
        
        # 计算TP, FP, FN
        tp = (binary_preds * labels).sum(dim=0)
        fp = (binary_preds * (1 - labels)).sum(dim=0)
        fn = ((1 - binary_preds) * labels).sum(dim=0)
        
        # 计算精确率和召回率
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 计算准确率
        accuracy = (binary_preds == labels).float().mean()
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item()
        }
    
    def _compute_per_attribute_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict:
        """计算每个属性的评估指标"""
        binary_preds = (preds > 0.5).float()
        
        metrics = {}
        for i, attr_name in enumerate(self.attr_names):
            # 获取当前属性的预测和标签
            attr_preds = binary_preds[:, i]
            attr_labels = labels[:, i]
            
            # 计算TP, FP, FN
            tp = (attr_preds * attr_labels).sum()
            fp = (attr_preds * (1 - attr_labels)).sum()
            fn = ((1 - attr_preds) * attr_labels).sum()
            
            # 计算指标
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            accuracy = (attr_preds == attr_labels).float().mean()
            
            metrics[attr_name] = {
                'accuracy': accuracy.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item()
            }
            
        return metrics
    
    def _generate_visualizations(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """生成评估可视化"""
        # 1. PR曲线
        self._plot_pr_curves(preds, labels, save_dir)
        
        # 2. 混淆矩阵
        self._plot_confusion_matrices(preds, labels, save_dir)
        
        # 3. 属性性能对比
        self._plot_attribute_performance(preds, labels, save_dir)
    
    def _plot_pr_curves(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制PR曲线"""
        plt.figure(figsize=(12, 8))
        
        # 为每个属性计算PR曲线
        for i, attr_name in enumerate(self.attr_names):
            precision, recall, _ = precision_recall_curve(
                labels[:, i].numpy(),
                preds[:, i].numpy()
            )
            ap = average_precision_score(labels[:, i].numpy(), preds[:, i].numpy())
            
            plt.plot(recall, precision, lw=2, label=f'{attr_name} (AP={ap:.2f})')
        
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('各属性的PR曲线')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pr_curves.png'))
        plt.close()
    
    def _plot_confusion_matrices(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制混淆矩阵"""
        binary_preds = (preds > 0.5).float()
        
        for i, attr_name in enumerate(self.attr_names):
            cm = confusion_matrix(labels[:, i].numpy(), binary_preds[:, i].numpy())
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'属性"{attr_name}"的混淆矩阵')
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'confusion_matrix_{attr_name}.png'))
            plt.close()
    
    def _plot_attribute_performance(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制属性性能对比图"""
        metrics = self._compute_per_attribute_metrics(preds, labels)
        
        # 准备数据
        attr_names = []
        accuracies = []
        f1_scores = []
        
        for attr_name, attr_metrics in metrics.items():
            attr_names.append(attr_name)
            accuracies.append(attr_metrics['accuracy'])
            f1_scores.append(attr_metrics['f1'])
        
        # 创建图表
        plt.figure(figsize=(15, 6))
        x = np.arange(len(attr_names))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='准确率')
        plt.bar(x + width/2, f1_scores, width, label='F1分数')
        
        plt.xlabel('属性')
        plt.ylabel('分数')
        plt.title('各属性的性能对比')
        plt.xticks(x, attr_names, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attribute_performance.png'))
        plt.close() 