import os
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, hamming_loss, multilabel_confusion_matrix
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple
import pandas as pd
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionMultiLabelEvaluator:
    """服饰多标签分类模型评估器，提供全面的评估指标和可视化"""
    
    def __init__(self, model, data_loader, device, attr_names, attr_types=None):
        """
        初始化评估器
        
        Args:
            model: 待评估的模型
            data_loader: 数据加载器
            device: 计算设备
            attr_names: 属性名称列表
            attr_types: 属性类型字典 {attr_name: type_id}，用于分组评估
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.attr_names = attr_names
        self.attr_types = attr_types or self._get_default_attr_types()
        
        # 属性类型定义
        self.type_definitions = {
            1: "纹理相关 (Texture)",
            2: "面料相关 (Fabric)", 
            3: "形状相关 (Shape)",
            4: "部件相关 (Part)",
            5: "风格相关 (Style)",
            6: "合身度 (Fit)"
        }
        
    def _get_default_attr_types(self):
        """获取默认属性类型映射"""
        # 基于DeepFashion数据集的属性类型定义
        default_types = {
            'floral': 1, 'graphic': 1, 'striped': 1, 'embroidered': 1, 'pleated': 1,
            'solid': 1, 'lattice': 1, 'long_sleeve': 2, 'short_sleeve': 2, 'sleeveless': 2,
            'maxi_length': 3, 'mini_length': 3, 'no_dress': 3, 'crew_neckline': 4,
            'v_neckline': 4, 'square_neckline': 4, 'no_neckline': 4, 'denim': 5,
            'chiffon': 5, 'cotton': 5, 'leather': 5, 'faux': 5, 'knit': 5,
            'tight': 6, 'loose': 6, 'conventional': 6
        }
        return default_types
    
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
        all_confidences = []
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
                
                # 获取预测结果和置信度
                preds = torch.sigmoid(outputs['attr_logits'])
                confidences = preds.clone()
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_confidences.append(confidences.cpu())
        
        # 合并所有预测和标签
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        
        # 计算平均推理时间
        avg_inference_time = np.mean(inference_times)
        
        # 计算各项指标
        metrics = {}
        
        # 1. 基础指标
        metrics['basic_metrics'] = self._compute_basic_metrics(all_preds, all_labels)
        
        # 2. 高级指标
        metrics['advanced_metrics'] = self._compute_advanced_metrics(all_preds, all_labels, all_confidences)
        
        # 3. 按属性类型分组的指标
        metrics['type_based_metrics'] = self._compute_type_based_metrics(all_preds, all_labels)
        
        # 4. 每个属性的详细指标
        metrics['per_attribute_metrics'] = self._compute_per_attribute_metrics(all_preds, all_labels, all_confidences)
        
        # 5. 标签不平衡分析
        metrics['imbalance_analysis'] = self._analyze_label_imbalance(all_labels)
        
        # 6. 置信度校准分析
        metrics['calibration_analysis'] = self._analyze_calibration(all_confidences, all_labels)
        
        # 7. 性能指标
        metrics['performance_metrics'] = {
            'avg_inference_time': avg_inference_time,
            'throughput': 1.0 / avg_inference_time if avg_inference_time > 0 else 0
        }
        
        # 生成可视化
        self._generate_comprehensive_visualizations(all_preds, all_labels, all_confidences, save_dir)
        
        return metrics
    
    def _compute_basic_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict:
        """计算基础评估指标"""
        # 使用0.5作为阈值
        binary_preds = (preds > 0.5).float()
        
        # 计算TP, FP, FN
        tp = (binary_preds * labels).sum(dim=0)
        fp = (binary_preds * (1 - labels)).sum(dim=0)
        fn = ((1 - binary_preds) * labels).sum(dim=0)
        tn = ((1 - binary_preds) * (1 - labels)).sum(dim=0)
        
        # 计算精确率和召回率
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # 计算准确率
        accuracy = (binary_preds == labels).float().mean()
        
        # 计算Hamming Loss
        hamming_loss_score = hamming_loss(labels.numpy(), binary_preds.numpy())
        
        # 计算子集准确率 (Exact Match Ratio)
        subset_accuracy = (binary_preds == labels).all(dim=1).float().mean()
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item(),
            'hamming_loss': hamming_loss_score,
            'subset_accuracy': subset_accuracy.item(),
            'micro_precision': precision.mean().item(),
            'micro_recall': recall.mean().item(),
            'micro_f1': f1.mean().item()
        }
    
    def _compute_advanced_metrics(self, preds: torch.Tensor, labels: torch.Tensor, confidences: torch.Tensor) -> Dict:
        """计算高级评估指标"""
        # ROC AUC
        roc_auc_scores = []
        for i in range(preds.shape[1]):
            try:
                auc = roc_auc_score(labels[:, i].numpy(), confidences[:, i].numpy())
                roc_auc_scores.append(auc)
            except ValueError:
                roc_auc_scores.append(0.5)  # 如果只有一个类别，设为0.5
        
        # Average Precision Score
        ap_scores = []
        for i in range(preds.shape[1]):
            try:
                ap = average_precision_score(labels[:, i].numpy(), confidences[:, i].numpy())
                ap_scores.append(ap)
            except ValueError:
                ap_scores.append(0.0)
        
        # 计算标签覆盖率
        label_coverage = labels.sum(dim=0) / labels.shape[0]
        
        # 计算预测覆盖率
        binary_preds = (preds > 0.5).float()
        pred_coverage = binary_preds.sum(dim=0) / binary_preds.shape[0]
        
        return {
            'mean_roc_auc': np.mean(roc_auc_scores),
            'mean_ap': np.mean(ap_scores),
            'roc_auc_per_class': roc_auc_scores,
            'ap_per_class': ap_scores,
            'label_coverage': label_coverage.tolist(),
            'prediction_coverage': pred_coverage.tolist(),
            'coverage_ratio': (pred_coverage / (label_coverage + 1e-8)).tolist()
        }
    
    def _compute_type_based_metrics(self, preds: torch.Tensor, labels: torch.Tensor) -> Dict:
        """按属性类型计算指标"""
        type_metrics = {}
        
        # 按类型分组属性
        type_groups = defaultdict(list)
        for i, attr_name in enumerate(self.attr_names):
            attr_type = self.attr_types.get(attr_name, 0)
            type_groups[attr_type].append(i)
        
        for type_id, indices in type_groups.items():
            if not indices:
                continue
                
            type_preds = preds[:, indices]
            type_labels = labels[:, indices]
            
            # 计算该类型的指标
            binary_preds = (type_preds > 0.5).float()
            
            tp = (binary_preds * type_labels).sum()
            fp = (binary_preds * (1 - type_labels)).sum()
            fn = ((1 - binary_preds) * type_labels).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            accuracy = (binary_preds == type_labels).float().mean()
            
            type_name = self.type_definitions.get(type_id, f"Type_{type_id}")
            type_metrics[type_name] = {
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'accuracy': accuracy.item(),
                'num_attributes': len(indices)
            }
        
        return type_metrics
    
    def _compute_per_attribute_metrics(self, preds: torch.Tensor, labels: torch.Tensor, confidences: torch.Tensor) -> Dict:
        """计算每个属性的详细指标"""
        metrics = {}
        
        for i, attr_name in enumerate(self.attr_names):
            attr_preds = preds[:, i]
            attr_labels = labels[:, i]
            attr_confidences = confidences[:, i]
            
            # 基础指标
            binary_preds = (attr_preds > 0.5).float()
            
            tp = (binary_preds * attr_labels).sum()
            fp = (binary_preds * (1 - attr_labels)).sum()
            fn = ((1 - binary_preds) * attr_labels).sum()
            tn = ((1 - binary_preds) * (1 - attr_labels)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            accuracy = (binary_preds == attr_labels).float().mean()
            
            # 高级指标
            try:
                roc_auc = roc_auc_score(attr_labels.numpy(), attr_confidences.numpy())
            except ValueError:
                roc_auc = 0.5
                
            try:
                ap = average_precision_score(attr_labels.numpy(), attr_confidences.numpy())
            except ValueError:
                ap = 0.0
            
            # 标签分布
            positive_ratio = attr_labels.mean().item()
            
            metrics[attr_name] = {
                'accuracy': accuracy.item(),
                'precision': precision.item(),
                'recall': recall.item(),
                'f1': f1.item(),
                'roc_auc': roc_auc,
                'ap': ap,
                'positive_ratio': positive_ratio,
                'support': attr_labels.sum().item()
            }
            
        return metrics
    
    def _analyze_label_imbalance(self, labels: torch.Tensor) -> Dict:
        """分析标签不平衡情况"""
        positive_ratios = labels.mean(dim=0)
        
        # 计算不平衡统计
        imbalance_stats = {
            'mean_positive_ratio': positive_ratios.mean().item(),
            'std_positive_ratio': positive_ratios.std().item(),
            'min_positive_ratio': positive_ratios.min().item(),
            'max_positive_ratio': positive_ratios.max().item(),
            'highly_imbalanced_attrs': [],  # 正样本比例 < 0.1 的属性
            'balanced_attrs': [],  # 0.1 <= 正样本比例 <= 0.9 的属性
            'positive_ratios': positive_ratios.tolist()
        }
        
        for i, ratio in enumerate(positive_ratios):
            attr_name = self.attr_names[i] if i < len(self.attr_names) else f"attr_{i}"
            if ratio < 0.1:
                imbalance_stats['highly_imbalanced_attrs'].append({
                    'name': attr_name,
                    'positive_ratio': ratio.item()
                })
            elif 0.1 <= ratio <= 0.9:
                imbalance_stats['balanced_attrs'].append({
                    'name': attr_name,
                    'positive_ratio': ratio.item()
                })
        
        return imbalance_stats
    
    def _analyze_calibration(self, confidences: torch.Tensor, labels: torch.Tensor) -> Dict:
        """分析置信度校准"""
        calibration_metrics = {}
        
        for i, attr_name in enumerate(self.attr_names):
            if i >= confidences.shape[1]:
                break
                
            attr_confidences = confidences[:, i]
            attr_labels = labels[:, i]
            
            try:
                # 计算校准曲线
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    attr_labels.numpy(), attr_confidences.numpy(), n_bins=10
                )
                
                # 计算ECE (Expected Calibration Error)
                bin_boundaries = np.linspace(0, 1, 11)
                bin_lowers = bin_boundaries[:-1]
                bin_uppers = bin_boundaries[1:]
                
                ece = 0.0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (attr_confidences > bin_lower) & (attr_confidences <= bin_upper)
                    bin_size = in_bin.sum()
                    if bin_size > 0:
                        bin_accuracy = attr_labels[in_bin].float().mean()
                        bin_confidence = attr_confidences[in_bin].mean()
                        ece += bin_size * abs(bin_accuracy - bin_confidence)
                
                ece /= len(attr_confidences)
                
                calibration_metrics[attr_name] = {
                    'ece': ece.item(),
                    'calibration_curve': {
                        'fraction_of_positives': fraction_of_positives.tolist(),
                        'mean_predicted_value': mean_predicted_value.tolist()
                    }
                }
                
            except Exception as e:
                logger.warning(f"计算属性 {attr_name} 的校准指标失败: {str(e)}")
                calibration_metrics[attr_name] = {
                    'ece': float('nan'),
                    'calibration_curve': {'fraction_of_positives': [], 'mean_predicted_value': []}
                }
        
        return calibration_metrics
    
    def _generate_comprehensive_visualizations(self, preds: torch.Tensor, labels: torch.Tensor, 
                                            confidences: torch.Tensor, save_dir: str):
        """生成全面的可视化图表"""
        # 1. PR曲线
        self._plot_pr_curves(preds, labels, save_dir)
        
        # 2. ROC曲线
        self._plot_roc_curves(confidences, labels, save_dir)
        
        # 3. 混淆矩阵
        self._plot_confusion_matrices(preds, labels, save_dir)
        
        # 4. 属性性能对比
        self._plot_attribute_performance(preds, labels, save_dir)
        
        # 5. 类型分组性能
        self._plot_type_based_performance(preds, labels, save_dir)
        
        # 6. 标签不平衡分析
        self._plot_label_imbalance_analysis(labels, save_dir)
        
        # 7. 置信度校准图
        self._plot_calibration_curves(confidences, labels, save_dir)
        
        # 8. 预测覆盖率分析
        self._plot_coverage_analysis(preds, labels, save_dir)
    
    def _plot_pr_curves(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制PR曲线"""
        plt.figure(figsize=(15, 10))
        
        # 为每个属性计算PR曲线
        for i, attr_name in enumerate(self.attr_names):
            if i >= preds.shape[1]:
                break
            precision, recall, _ = precision_recall_curve(
                labels[:, i].numpy(),
                preds[:, i].numpy()
            )
            ap = average_precision_score(labels[:, i].numpy(), preds[:, i].numpy())
            
            plt.plot(recall, precision, lw=1, alpha=0.7, label=f'{attr_name} (AP={ap:.3f})')
        
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('各属性的PR曲线')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'pr_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, confidences: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制ROC曲线"""
        plt.figure(figsize=(15, 10))
        
        for i, attr_name in enumerate(self.attr_names):
            if i >= confidences.shape[1]:
                break
            try:
                fpr, tpr, _ = roc_curve(labels[:, i].numpy(), confidences[:, i].numpy())
                auc = roc_auc_score(labels[:, i].numpy(), confidences[:, i].numpy())
                plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f'{attr_name} (AUC={auc:.3f})')
            except ValueError:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('各属性的ROC曲线')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制混淆矩阵"""
        binary_preds = (preds > 0.5).float()
        
        # 选择前9个属性进行可视化
        num_attrs = min(9, preds.shape[1])
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()
        
        for i in range(num_attrs):
            attr_name = self.attr_names[i] if i < len(self.attr_names) else f"attr_{i}"
            cm = confusion_matrix(labels[:, i].numpy(), binary_preds[:, i].numpy())
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{attr_name}')
            axes[i].set_ylabel('真实标签')
            axes[i].set_xlabel('预测标签')
        
        # 隐藏多余的子图
        for i in range(num_attrs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attribute_performance(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制属性性能对比图"""
        metrics = self._compute_per_attribute_metrics(preds, labels, torch.sigmoid(preds))
        
        # 准备数据
        attr_names = []
        accuracies = []
        f1_scores = []
        roc_aucs = []
        
        for attr_name, attr_metrics in metrics.items():
            attr_names.append(attr_name)
            accuracies.append(attr_metrics['accuracy'])
            f1_scores.append(attr_metrics['f1'])
            roc_aucs.append(attr_metrics['roc_auc'])
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        x = np.arange(len(attr_names))
        width = 0.25
        
        # 准确率和F1分数
        ax1.bar(x - width, accuracies, width, label='准确率', alpha=0.8)
        ax1.bar(x, f1_scores, width, label='F1分数', alpha=0.8)
        ax1.bar(x + width, roc_aucs, width, label='ROC AUC', alpha=0.8)
        
        ax1.set_xlabel('属性')
        ax1.set_ylabel('分数')
        ax1.set_title('各属性的性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(attr_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 正样本比例
        positive_ratios = [metrics[name]['positive_ratio'] for name in attr_names]
        ax2.bar(x, positive_ratios, alpha=0.8, color='orange')
        ax2.set_xlabel('属性')
        ax2.set_ylabel('正样本比例')
        ax2.set_title('各属性的标签分布')
        ax2.set_xticks(x)
        ax2.set_xticklabels(attr_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'attribute_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_type_based_performance(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制类型分组性能图"""
        type_metrics = self._compute_type_based_metrics(preds, labels)
        
        if not type_metrics:
            return
        
        # 准备数据
        type_names = list(type_metrics.keys())
        f1_scores = [type_metrics[name]['f1'] for name in type_names]
        accuracies = [type_metrics[name]['accuracy'] for name in type_names]
        num_attrs = [type_metrics[name]['num_attributes'] for name in type_names]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # F1分数和准确率对比
        x = np.arange(len(type_names))
        width = 0.35
        
        ax1.bar(x - width/2, f1_scores, width, label='F1分数', alpha=0.8)
        ax1.bar(x + width/2, accuracies, width, label='准确率', alpha=0.8)
        
        ax1.set_xlabel('属性类型')
        ax1.set_ylabel('分数')
        ax1.set_title('各属性类型的性能对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(type_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 属性数量分布
        ax2.bar(x, num_attrs, alpha=0.8, color='green')
        ax2.set_xlabel('属性类型')
        ax2.set_ylabel('属性数量')
        ax2.set_title('各类型包含的属性数量')
        ax2.set_xticks(x)
        ax2.set_xticklabels(type_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'type_based_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_label_imbalance_analysis(self, labels: torch.Tensor, save_dir: str):
        """绘制标签不平衡分析图"""
        positive_ratios = labels.mean(dim=0)
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 正样本比例分布
        ax1.hist(positive_ratios.numpy(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(positive_ratios.mean(), color='red', linestyle='--', label=f'平均值: {positive_ratios.mean():.3f}')
        ax1.set_xlabel('正样本比例')
        ax1.set_ylabel('属性数量')
        ax1.set_title('标签不平衡分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 排序后的正样本比例
        sorted_ratios, sorted_indices = torch.sort(positive_ratios, descending=True)
        attr_names_sorted = [self.attr_names[i] if i < len(self.attr_names) else f"attr_{i}" 
                           for i in sorted_indices]
        
        ax2.bar(range(len(sorted_ratios)), sorted_ratios.numpy(), alpha=0.7, color='lightcoral')
        ax2.set_xlabel('属性索引')
        ax2.set_ylabel('正样本比例')
        ax2.set_title('各属性正样本比例排序')
        ax2.grid(True, alpha=0.3)
        
        # 添加一些关键属性的标签
        for i in range(0, len(sorted_ratios), max(1, len(sorted_ratios)//10)):
            ax2.annotate(attr_names_sorted[i], 
                        xy=(i, sorted_ratios[i]), 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', fontsize=8, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'label_imbalance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curves(self, confidences: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制置信度校准曲线"""
        # 选择前6个属性进行可视化
        num_attrs = min(6, confidences.shape[1])
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i in range(num_attrs):
            attr_name = self.attr_names[i] if i < len(self.attr_names) else f"attr_{i}"
            attr_confidences = confidences[:, i]
            attr_labels = labels[:, i]
            
            try:
                fraction_of_positives, mean_predicted_value = calibration_curve(
                    attr_labels.numpy(), attr_confidences.numpy(), n_bins=10
                )
                
                axes[i].plot(mean_predicted_value, fraction_of_positives, 'o-', label='校准曲线')
                axes[i].plot([0, 1], [0, 1], 'k--', label='完美校准')
                axes[i].set_xlabel('平均预测概率')
                axes[i].set_ylabel('实际正样本比例')
                axes[i].set_title(f'{attr_name}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                axes[i].text(0.5, 0.5, f'无法计算\n{attr_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{attr_name}')
        
        # 隐藏多余的子图
        for i in range(num_attrs, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'calibration_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_coverage_analysis(self, preds: torch.Tensor, labels: torch.Tensor, save_dir: str):
        """绘制预测覆盖率分析图"""
        binary_preds = (preds > 0.5).float()
        
        # 计算标签覆盖率和预测覆盖率
        label_coverage = labels.sum(dim=0) / labels.shape[0]
        pred_coverage = binary_preds.sum(dim=0) / binary_preds.shape[0]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 覆盖率对比
        x = np.arange(len(self.attr_names))
        width = 0.35
        
        ax1.bar(x - width/2, label_coverage.numpy(), width, label='标签覆盖率', alpha=0.8)
        ax1.bar(x + width/2, pred_coverage.numpy(), width, label='预测覆盖率', alpha=0.8)
        
        ax1.set_xlabel('属性')
        ax1.set_ylabel('覆盖率')
        ax1.set_title('标签覆盖率 vs 预测覆盖率')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.attr_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 覆盖率差异
        coverage_diff = pred_coverage - label_coverage
        colors = ['red' if diff > 0 else 'blue' for diff in coverage_diff]
        ax2.bar(x, coverage_diff.numpy(), color=colors, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('属性')
        ax2.set_ylabel('覆盖率差异 (预测 - 标签)')
        ax2.set_title('预测覆盖率与标签覆盖率的差异')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.attr_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'coverage_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_evaluation_report(self, metrics: Dict, save_dir: str = "evaluation_results") -> str:
        """生成详细的评估报告"""
        report_path = os.path.join(save_dir, "evaluation_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("服饰多标签分类模型评估报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 基础指标
            f.write("1. 基础评估指标\n")
            f.write("-" * 40 + "\n")
            basic = metrics['basic_metrics']
            f.write(f"准确率: {basic['accuracy']:.4f}\n")
            f.write(f"精确率: {basic['precision']:.4f}\n")
            f.write(f"召回率: {basic['recall']:.4f}\n")
            f.write(f"F1分数: {basic['f1']:.4f}\n")
            f.write(f"Hamming Loss: {basic['hamming_loss']:.4f}\n")
            f.write(f"子集准确率: {basic['subset_accuracy']:.4f}\n\n")
            
            # 高级指标
            f.write("2. 高级评估指标\n")
            f.write("-" * 40 + "\n")
            advanced = metrics['advanced_metrics']
            f.write(f"平均ROC AUC: {advanced['mean_roc_auc']:.4f}\n")
            f.write(f"平均AP: {advanced['mean_ap']:.4f}\n\n")
            
            # 类型分组指标
            f.write("3. 按属性类型的性能\n")
            f.write("-" * 40 + "\n")
            type_metrics = metrics['type_based_metrics']
            for type_name, type_metric in type_metrics.items():
                f.write(f"{type_name}:\n")
                f.write(f"  F1分数: {type_metric['f1']:.4f}\n")
                f.write(f"  准确率: {type_metric['accuracy']:.4f}\n")
                f.write(f"  属性数量: {type_metric['num_attributes']}\n\n")
            
            # 标签不平衡分析
            f.write("4. 标签不平衡分析\n")
            f.write("-" * 40 + "\n")
            imbalance = metrics['imbalance_analysis']
            f.write(f"平均正样本比例: {imbalance['mean_positive_ratio']:.4f}\n")
            f.write(f"正样本比例标准差: {imbalance['std_positive_ratio']:.4f}\n")
            f.write(f"最小正样本比例: {imbalance['min_positive_ratio']:.4f}\n")
            f.write(f"最大正样本比例: {imbalance['max_positive_ratio']:.4f}\n")
            f.write(f"高度不平衡属性数量: {len(imbalance['highly_imbalanced_attrs'])}\n")
            f.write(f"平衡属性数量: {len(imbalance['balanced_attrs'])}\n\n")
            
            # 性能指标
            f.write("5. 性能指标\n")
            f.write("-" * 40 + "\n")
            perf = metrics['performance_metrics']
            f.write(f"平均推理时间: {perf['avg_inference_time']:.4f}秒\n")
            f.write(f"吞吐量: {perf['throughput']:.2f}样本/秒\n\n")
            
            # 最佳和最差属性
            f.write("6. 属性性能排名\n")
            f.write("-" * 40 + "\n")
            per_attr = metrics['per_attribute_metrics']
            
            # 按F1分数排序
            sorted_attrs = sorted(per_attr.items(), key=lambda x: x[1]['f1'], reverse=True)
            
            f.write("最佳性能属性 (按F1分数):\n")
            for i, (attr_name, attr_metric) in enumerate(sorted_attrs[:5]):
                f.write(f"  {i+1}. {attr_name}: F1={attr_metric['f1']:.4f}, "
                       f"AP={attr_metric['ap']:.4f}, 正样本比例={attr_metric['positive_ratio']:.4f}\n")
            
            f.write("\n最差性能属性 (按F1分数):\n")
            for i, (attr_name, attr_metric) in enumerate(sorted_attrs[-5:]):
                f.write(f"  {i+1}. {attr_name}: F1={attr_metric['f1']:.4f}, "
                       f"AP={attr_metric['ap']:.4f}, 正样本比例={attr_metric['positive_ratio']:.4f}\n")
        
        return report_path


# 兼容性包装器，保持与原有代码的兼容性
class ModelEvaluator(FashionMultiLabelEvaluator):
    """向后兼容的模型评估器"""
    
    def __init__(self, model, data_loader, device, attr_names):
        super().__init__(model, data_loader, device, attr_names)
    
    def evaluate(self, save_dir: str = "evaluation_results") -> Dict:
        """执行评估并返回兼容格式的结果"""
        metrics = super().evaluate(save_dir)
        
        # 返回兼容格式的结果
        return {
            'accuracy': metrics['basic_metrics']['accuracy'],
            'precision': metrics['basic_metrics']['precision'],
            'recall': metrics['basic_metrics']['recall'],
            'f1': metrics['basic_metrics']['f1'],
            'per_attribute': metrics['per_attribute_metrics']
        }