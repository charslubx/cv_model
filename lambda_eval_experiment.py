"""
λ 超参数评估实验（基于已训练模型）
用于测试不同 λ 值对已训练模型性能的影响
不需要重新训练，直接在验证集上评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LambdaEvaluationExperiment:
    """
    λ 超参数评估实验类
    基于已训练好的模型，测试不同 λ 值对图构建和性能的影响
    """
    
    def __init__(self, pretrained_model_path, val_loader, device, num_classes=26):
        """
        初始化 λ 评估实验
        
        Args:
            pretrained_model_path: 预训练模型路径
            val_loader: 验证数据加载器
            device: 计算设备
            num_classes: 类别数量
        """
        self.pretrained_model_path = pretrained_model_path
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # 要测试的 λ 值
        self.lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        # 加载预训练模型
        logger.info(f"加载预训练模型: {pretrained_model_path}")
        self.base_model = torch.load(pretrained_model_path, map_location=device)
        self.base_model.eval()
        
        logger.info(f"模型加载成功！")
        logger.info(f"测试 λ 值: {self.lambda_values}")
    
    def _modify_lambda_in_forward(self, model, lambda_val):
        """
        动态修改模型forward方法中的λ值
        通过monkey patching实现
        """
        original_forward = model.forward
        
        def new_forward(images, img_names=None):
            # 提取特征
            features = model.feature_extractor(images)
            
            if torch.isnan(features['global']).any():
                features['global'] = torch.nan_to_num(features['global'], nan=0.0)
            
            outputs = {}
            
            # 特征增强
            global_features = model.feature_enhancer(features['global'])
            global_features = torch.nn.functional.normalize(global_features, p=2, dim=1)
            
            # 构建邻接矩阵（使用指定的λ值）
            sim_matrix = torch.nn.functional.cosine_similarity(
                global_features.unsqueeze(1),
                global_features.unsqueeze(0),
                dim=2
            )
            
            # 使用指定的λ值计算动态阈值
            mean_sim = sim_matrix.mean()
            std_sim = sim_matrix.std()
            threshold = mean_sim + lambda_val * std_sim  # 使用指定的λ
            
            # 构建稀疏邻接矩阵
            adj_matrix = (sim_matrix > threshold).float()
            adj_matrix = adj_matrix * sim_matrix
            
            if adj_matrix.sum() == 0:
                adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
            
            # GAT特征增强
            gat_features = model.gat(global_features, adj_matrix)
            gat_features = torch.nn.functional.normalize(gat_features, p=2, dim=1)
            
            # GCN分类
            edge_index = adj_matrix.nonzero().t()
            gcn_logits = model.gcn(gat_features, edge_index)
            
            # CNN特征分类
            cnn_logits = model.attr_head(global_features)
            
            # 预测类别权重
            class_weights = model.class_weight_predictor(global_features)
            class_weights = torch.clamp(class_weights, 0.1, 1.0)
            
            # 特征融合
            fusion_input = torch.cat([global_features, gat_features], dim=1)
            fusion_weights = model.fusion_gate(fusion_input)
            
            # 加权融合
            attr_logits = fusion_weights[:, 0:1] * gcn_logits + fusion_weights[:, 1:2] * cnn_logits
            attr_logits = attr_logits * class_weights
            attr_logits = torch.clamp(attr_logits, -10, 10)
            
            outputs['attr_logits'] = attr_logits
            outputs['class_weights'] = class_weights
            
            # 存储图统计信息
            outputs['graph_stats'] = {
                'num_edges': adj_matrix.sum().item() / 2,  # 无向图，除以2
                'avg_degree': adj_matrix.sum(dim=1).mean().item(),
                'threshold': threshold.item(),
                'mean_sim': mean_sim.item(),
                'std_sim': std_sim.item()
            }
            
            return outputs
        
        model.forward = new_forward
        return model
    
    def evaluate_with_lambda(self, lambda_val):
        """使用指定的λ值评估模型"""
        logger.info(f"\n{'='*80}")
        logger.info(f"评估 λ = {lambda_val}")
        logger.info(f"{'='*80}\n")
        
        # 修改模型的forward方法以使用指定的λ
        model = self._modify_lambda_in_forward(self.base_model, lambda_val)
        model.eval()
        
        all_preds = []
        all_targets = []
        graph_stats_list = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'评估 λ={lambda_val}'):
                images = batch['image'].to(self.device)
                labels = batch['attr_labels'].to(self.device)
                
                outputs = model(images)
                
                all_preds.append(torch.sigmoid(outputs['attr_logits']).cpu())
                all_targets.append(labels.cpu())
                
                # 收集图统计信息
                if 'graph_stats' in outputs:
                    graph_stats_list.append(outputs['graph_stats'])
        
        # 计算指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._calculate_metrics(all_preds, all_targets)
        
        # 平均图统计信息
        if graph_stats_list:
            avg_graph_stats = {
                'num_edges': sum(s['num_edges'] for s in graph_stats_list) / len(graph_stats_list),
                'avg_degree': sum(s['avg_degree'] for s in graph_stats_list) / len(graph_stats_list),
                'threshold': sum(s['threshold'] for s in graph_stats_list) / len(graph_stats_list),
                'mean_sim': sum(s['mean_sim'] for s in graph_stats_list) / len(graph_stats_list),
                'std_sim': sum(s['std_sim'] for s in graph_stats_list) / len(graph_stats_list)
            }
            metrics['graph_stats'] = avg_graph_stats
        
        logger.info(f"\nλ = {lambda_val} 评估结果:")
        logger.info(f"  F1: {metrics['f1']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        if 'graph_stats' in metrics:
            logger.info(f"  图统计:")
            logger.info(f"    平均边数: {metrics['graph_stats']['num_edges']:.1f}")
            logger.info(f"    平均度数: {metrics['graph_stats']['avg_degree']:.2f}")
            logger.info(f"    阈值: {metrics['graph_stats']['threshold']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, predictions, targets, threshold=0.5):
        """计算评估指标"""
        binary_preds = (predictions > threshold).float()
        
        tp = (binary_preds * targets).sum(dim=0)
        fp = (binary_preds * (1 - targets)).sum(dim=0)
        fn = ((1 - binary_preds) * targets).sum(dim=0)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        return {
            'precision': precision.mean().item(),
            'recall': recall.mean().item(),
            'f1': f1.mean().item()
        }
    
    def run_experiment(self, save_dir='lambda_eval_results'):
        """运行完整的 λ 评估实验"""
        os.makedirs(save_dir, exist_ok=True)
        all_results = {}
        
        logger.info("\n" + "="*80)
        logger.info("开始 λ 超参数评估实验")
        logger.info(f"基于模型: {self.pretrained_model_path}")
        logger.info("="*80)
        
        for lambda_val in self.lambda_values:
            metrics = self.evaluate_with_lambda(lambda_val)
            
            result = {
                'lambda': lambda_val,
                'f1': metrics['f1'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            }
            
            if 'graph_stats' in metrics:
                result['graph_stats'] = metrics['graph_stats']
            
            all_results[f'lambda_{lambda_val}'] = result
            
            # 保存单个λ的结果
            result_file = os.path.join(save_dir, f'lambda_{lambda_val}_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
        
        # 保存汇总结果
        self._save_summary(all_results, save_dir)
        
        # 可视化结果
        self._visualize_results(all_results, save_dir)
        
        logger.info("\n" + "="*80)
        logger.info("λ 超参数评估实验完成！")
        logger.info("="*80)
        
        return all_results
    
    def _save_summary(self, results, save_dir):
        """保存实验汇总"""
        summary_data = []
        
        for key, result in results.items():
            row = {
                'lambda': result['lambda'],
                'f1': result['f1'],
                'precision': result['precision'],
                'recall': result['recall']
            }
            
            if 'graph_stats' in result:
                row['avg_edges'] = result['graph_stats']['num_edges']
                row['avg_degree'] = result['graph_stats']['avg_degree']
                row['threshold'] = result['graph_stats']['threshold']
            
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('lambda')
        
        csv_path = os.path.join(save_dir, 'lambda_eval_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info("\n" + "="*80)
        logger.info("λ 超参数评估汇总")
        logger.info("="*80)
        logger.info("\n" + df.to_string(index=False))
        
        # 找出最佳 λ
        best_lambda = df.loc[df['f1'].idxmax(), 'lambda']
        best_f1 = df['f1'].max()
        
        logger.info("\n" + "="*80)
        logger.info(f"最佳 λ 值: {best_lambda}")
        logger.info(f"最佳 F1 分数: {best_f1:.4f}")
        logger.info("="*80)
        
        # 保存最佳 λ 信息
        best_lambda_info = {
            'best_lambda': float(best_lambda),
            'best_f1': float(best_f1),
            'pretrained_model': self.pretrained_model_path,
            'conclusion': f"基于已训练模型的评估表明，当 λ 取 {best_lambda} 时，"
                         f"在验证集上可获得最优的 F1 表现（{best_f1:.4f}）。"
                         f"过小的 λ 导致图过于稠密，引入噪声邻居；"
                         f"过大的 λ 又会使图过于稀疏，削弱邻域信息传递。"
        }
        
        with open(os.path.join(save_dir, 'best_lambda.json'), 'w', encoding='utf-8') as f:
            json.dump(best_lambda_info, f, indent=4, ensure_ascii=False)
    
    def _visualize_results(self, results, save_dir):
        """可视化实验结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        lambda_vals = [results[k]['lambda'] for k in sorted(results.keys())]
        f1_scores = [results[k]['f1'] for k in sorted(results.keys())]
        precisions = [results[k]['precision'] for k in sorted(results.keys())]
        recalls = [results[k]['recall'] for k in sorted(results.keys())]
        
        # 1. 性能指标曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 左图：F1, Precision, Recall
        ax1.plot(lambda_vals, f1_scores, marker='o', label='F1 Score', linewidth=2, markersize=8)
        ax1.plot(lambda_vals, precisions, marker='s', label='Precision', linewidth=2, markersize=8)
        ax1.plot(lambda_vals, recalls, marker='^', label='Recall', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (λ)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Lambda vs Performance Metrics', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 右图：图统计信息（如果有）
        if 'graph_stats' in results[list(results.keys())[0]]:
            avg_edges = [results[k]['graph_stats']['num_edges'] for k in sorted(results.keys())]
            avg_degrees = [results[k]['graph_stats']['avg_degree'] for k in sorted(results.keys())]
            
            ax2_twin = ax2.twinx()
            line1 = ax2.plot(lambda_vals, avg_edges, marker='o', color='blue', 
                           label='Avg Edges', linewidth=2, markersize=8)
            line2 = ax2_twin.plot(lambda_vals, avg_degrees, marker='s', color='red', 
                                 label='Avg Degree', linewidth=2, markersize=8)
            
            ax2.set_xlabel('Lambda (λ)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Average Edges', fontsize=12, fontweight='bold', color='blue')
            ax2_twin.set_ylabel('Average Degree', fontsize=12, fontweight='bold', color='red')
            ax2.set_title('Lambda vs Graph Properties', fontsize=14, fontweight='bold')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2_twin.tick_params(axis='y', labelcolor='red')
            ax2.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='best')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'lambda_evaluation.png'), dpi=300)
        plt.close()
        
        logger.info(f"\n可视化结果已保存到: {save_dir}")


def main():
    """主函数"""
    from dataset import create_data_loaders
    from experiment_config import EXPERIMENT_CONFIG
    
    logger.info("\n" + "="*80)
    logger.info("λ 超参数评估实验（基于预训练模型）")
    logger.info("="*80)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n使用设备: {device}")
    
    # 创建数据加载器
    logger.info("\n创建数据加载器...")
    _, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    
    # 预训练模型路径
    pretrained_model_path = 'smart_mixed_checkpoints/best_model.pth'
    
    if not os.path.exists(pretrained_model_path):
        logger.error(f"找不到预训练模型: {pretrained_model_path}")
        logger.error("请先训练模型或指定正确的模型路径")
        return 1
    
    # 创建实验
    experiment = LambdaEvaluationExperiment(
        pretrained_model_path=pretrained_model_path,
        val_loader=val_loader,
        device=device,
        num_classes=26
    )
    
    # 运行实验
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'lambda_eval_results_{timestamp}'
    results = experiment.run_experiment(save_dir=save_dir)
    
    logger.info(f"\n实验完成！结果保存在: {save_dir}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

