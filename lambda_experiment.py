"""
λ 超参数实验框架
用于论文第四章的超参数分析实验
测试 λ = 0, 0.3, 0.5, 0.7 对模型性能的影响
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
from ablation_models import CNNWithAdaptiveGAT

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LambdaExperiment:
    """
    λ 超参数实验类
    用于测试动态阈值 τ = μ + λσ 中 λ 参数对模型性能的影响
    """
    
    def __init__(self, train_loader, val_loader, device, num_classes=26):
        """
        初始化 λ 实验
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 计算设备
            num_classes: 类别数量
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # 要测试的 λ 值
        self.lambda_values = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        logger.info(f"初始化 λ 超参数实验，测试值: {self.lambda_values}")
    
    def train_epoch(self, model, optimizer, criterion, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'训练 Epoch {epoch}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['attr_labels'].to(self.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs['attr_logits'], labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            all_preds.append(torch.sigmoid(outputs['attr_logits']).detach().cpu())
            all_targets.append(labels.cpu())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # 计算训练指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self, model, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='验证'):
                images = batch['image'].to(self.device)
                labels = batch['attr_labels'].to(self.device)
                
                outputs = model(images)
                loss = criterion(outputs['attr_logits'], labels)
                
                total_loss += loss.item()
                all_preds.append(torch.sigmoid(outputs['attr_logits']).cpu())
                all_targets.append(labels.cpu())
        
        # 计算验证指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _calculate_metrics(self, predictions, targets, threshold=0.5):
        """计算评估指标"""
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
    
    def run_experiment(self, epochs=30, learning_rate=3e-4, save_dir='lambda_results'):
        """
        运行完整的 λ 超参数实验
        
        Args:
            epochs: 训练轮数
            learning_rate: 学习率
            save_dir: 结果保存目录
        
        Returns:
            dict: 包含所有 λ 值的实验结果
        """
        os.makedirs(save_dir, exist_ok=True)
        all_results = {}
        
        logger.info("="*80)
        logger.info("开始 λ 超参数实验")
        logger.info("="*80)
        
        for lambda_val in self.lambda_values:
            logger.info(f"\n{'='*80}")
            logger.info(f"测试 λ = {lambda_val}")
            logger.info(f"{'='*80}\n")
            
            # 创建模型
            model = CNNWithAdaptiveGAT(
                num_classes=self.num_classes,
                cnn_type='resnet50',
                weights='IMAGENET1K_V1',
                lambda_threshold=lambda_val
            ).to(self.device)
            
            # 优化器和损失函数
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )
            criterion = nn.BCEWithLogitsLoss()
            
            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, verbose=True
            )
            
            # 训练和验证
            best_f1 = 0
            best_epoch = 0
            history = {'train': [], 'val': []}
            
            for epoch in range(1, epochs + 1):
                logger.info(f"\nλ = {lambda_val}, Epoch {epoch}/{epochs}")
                
                # 训练
                train_metrics = self.train_epoch(model, optimizer, criterion, epoch)
                history['train'].append({
                    'epoch': epoch,
                    **train_metrics
                })
                
                # 验证
                val_metrics = self.validate(model, criterion)
                history['val'].append({
                    'epoch': epoch,
                    **val_metrics
                })
                
                # 更新学习率
                scheduler.step(val_metrics['f1'])
                
                # 记录最佳结果
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    best_epoch = epoch
                    
                    # 保存最佳模型
                    model_save_path = os.path.join(save_dir, f'best_model_lambda_{lambda_val}.pth')
                    torch.save(model.state_dict(), model_save_path)
                
                logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                           f"F1: {train_metrics['f1']:.4f}, "
                           f"Precision: {train_metrics['precision']:.4f}, "
                           f"Recall: {train_metrics['recall']:.4f}")
                logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                           f"F1: {val_metrics['f1']:.4f}, "
                           f"Precision: {val_metrics['precision']:.4f}, "
                           f"Recall: {val_metrics['recall']:.4f}")
            
            # 训练结束后，加载最佳模型进行最终评估
            logger.info(f"\n加载最佳模型 (Epoch {best_epoch}) 进行最终评估...")
            best_model_path = os.path.join(save_dir, f'best_model_lambda_{lambda_val}.pth')
            model.load_state_dict(torch.load(best_model_path))
            
            # 使用最佳模型重新评估
            final_val_metrics = self.validate(model, criterion)
            
            logger.info(f"最佳模型最终验证结果:")
            logger.info(f"  Loss: {final_val_metrics['loss']:.4f}")
            logger.info(f"  F1: {final_val_metrics['f1']:.4f}")
            logger.info(f"  Precision: {final_val_metrics['precision']:.4f}")
            logger.info(f"  Recall: {final_val_metrics['recall']:.4f}")
            
            # 保存该 λ 值的结果
            lambda_result = {
                'lambda': lambda_val,
                'best_f1': best_f1,
                'best_epoch': best_epoch,
                'final_val_metrics': final_val_metrics,  # 使用最佳模型的评估结果
                'history': history
            }
            all_results[f'lambda_{lambda_val}'] = lambda_result
            
            logger.info(f"\nλ = {lambda_val} 实验完成")
            logger.info(f"最佳 F1: {best_f1:.4f} (Epoch {best_epoch})")
            
            # 保存单个 λ 的详细结果
            result_file = os.path.join(save_dir, f'lambda_{lambda_val}_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                # 转换 history 为可序列化格式
                serializable_result = {
                    'lambda': lambda_val,
                    'best_f1': float(best_f1),
                    'best_epoch': int(best_epoch),
                    'final_val_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                         for k, v in final_val_metrics.items()}
                }
                json.dump(serializable_result, f, indent=4, ensure_ascii=False)
        
        # 保存汇总结果
        self._save_summary(all_results, save_dir)
        
        # 可视化结果
        self._visualize_results(all_results, save_dir)
        
        logger.info("\n" + "="*80)
        logger.info("λ 超参数实验完成！")
        logger.info("="*80)
        
        return all_results
    
    def _save_summary(self, results, save_dir):
        """保存实验汇总"""
        summary_data = []
        
        for key, result in results.items():
            summary_data.append({
                'lambda': result['lambda'],
                'best_f1': result['best_f1'],
                'best_epoch': result['best_epoch'],
                'final_precision': result['final_val_metrics']['precision'],
                'final_recall': result['final_val_metrics']['recall'],
                'final_loss': result['final_val_metrics']['loss']
            })
        
        # 创建 DataFrame
        df = pd.DataFrame(summary_data)
        df = df.sort_values('lambda')
        
        # 保存为 CSV
        csv_path = os.path.join(save_dir, 'lambda_experiment_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 打印汇总
        logger.info("\n" + "="*80)
        logger.info("λ 超参数实验汇总")
        logger.info("="*80)
        logger.info("\n" + df.to_string(index=False))
        
        # 找出最佳 λ
        best_lambda = df.loc[df['best_f1'].idxmax(), 'lambda']
        best_f1 = df['best_f1'].max()
        
        logger.info("\n" + "="*80)
        logger.info(f"最佳 λ 值: {best_lambda}")
        logger.info(f"最佳 F1 分数: {best_f1:.4f}")
        logger.info("="*80)
        
        # 保存最佳 λ 信息
        best_lambda_info = {
            'best_lambda': float(best_lambda),
            'best_f1': float(best_f1),
            'conclusion': f"实验表明，当 λ 取 {best_lambda} 时，在验证集上可获得最优的 F1 表现；"
                         f"过小的 λ 导致图过于稠密，引入噪声邻居，"
                         f"过大的 λ 又会使图过于稀疏，削弱邻域信息传递。"
                         f"因此后续实验均固定 λ = {best_lambda}。"
        }
        
        with open(os.path.join(save_dir, 'best_lambda.json'), 'w', encoding='utf-8') as f:
            json.dump(best_lambda_info, f, indent=4, ensure_ascii=False)
    
    def _visualize_results(self, results, save_dir):
        """可视化实验结果"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. λ vs F1 曲线
        lambda_vals = [results[k]['lambda'] for k in sorted(results.keys())]
        best_f1s = [results[k]['best_f1'] for k in sorted(results.keys())]
        
        plt.figure(figsize=(10, 6))
        plt.plot(lambda_vals, best_f1s, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Lambda (λ)', fontsize=12)
        plt.ylabel('Best F1 Score', fontsize=12)
        plt.title('Impact of Lambda on Model Performance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'lambda_vs_f1.png'), dpi=300)
        plt.close()
        
        # 2. 训练曲线对比
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for key in sorted(results.keys()):
            result = results[key]
            lambda_val = result['lambda']
            history = result['history']
            
            # Loss 曲线
            train_losses = [h['loss'] for h in history['train']]
            val_losses = [h['loss'] for h in history['val']]
            epochs = list(range(1, len(train_losses) + 1))
            
            axes[0, 0].plot(epochs, train_losses, label=f'λ={lambda_val} (train)', alpha=0.7)
            axes[0, 1].plot(epochs, val_losses, label=f'λ={lambda_val} (val)', alpha=0.7)
            
            # F1 曲线
            train_f1s = [h['f1'] for h in history['train']]
            val_f1s = [h['f1'] for h in history['val']]
            
            axes[1, 0].plot(epochs, train_f1s, label=f'λ={lambda_val} (train)', alpha=0.7)
            axes[1, 1].plot(epochs, val_f1s, label=f'λ={lambda_val} (val)', alpha=0.7)
        
        # 设置子图
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training F1 Score', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation F1 Score', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves_comparison.png'), dpi=300)
        plt.close()
        
        # 3. 指标对比热力图
        metrics_data = []
        for key in sorted(results.keys()):
            result = results[key]
            metrics_data.append({
                'Lambda': result['lambda'],
                'Best F1': result['best_f1'],
                'Precision': result['final_val_metrics']['precision'],
                'Recall': result['final_val_metrics']['recall'],
                'Final Loss': result['final_val_metrics']['loss']
            })
        
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics = df_metrics.set_index('Lambda')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_metrics.T, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Score'})
        plt.title('Lambda Parameter Comparison Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Lambda (λ)', fontsize=12)
        plt.ylabel('Metrics', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'lambda_metrics_heatmap.png'), dpi=300)
        plt.close()
        
        logger.info(f"\n可视化结果已保存到: {save_dir}")

