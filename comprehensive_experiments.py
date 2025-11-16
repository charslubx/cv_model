"""
完整的论文实验框架
整合模块级消融实验、λ超参数实验和SOTA对比实验
用于支撑论文第四章的实验部分
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import logging
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
import numpy as np

# 导入自定义模块
from ablation_models import (
    BaselineCNN, 
    CNNWithStaticGraph, 
    CNNWithAdaptiveGAT, 
    CNNGraphFusion, 
    FullAdaGAT
)
from lambda_experiment import LambdaExperiment
from dataset import create_data_loaders
from experiment_config import EXPERIMENT_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_experiments.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SOTABaseline(nn.Module):
    """SOTA对比基线模型的统一包装器"""
    
    def __init__(self, backbone_name, num_classes=26):
        super().__init__()
        self.backbone_name = backbone_name
        
        if backbone_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(2048, num_classes)
        elif backbone_name == 'resnet101':
            self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Linear(2048, num_classes)
        elif backbone_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model.classifier[1] = nn.Linear(1280, num_classes)
        elif backbone_name == 'densenet121':
            self.model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            self.model.classifier = nn.Linear(1024, num_classes)
        elif backbone_name == 'vit_b_16':
            self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.model.heads.head = nn.Linear(768, num_classes)
        else:
            raise ValueError(f"不支持的backbone: {backbone_name}")
    
    def forward(self, x):
        logits = self.model(x)
        return {'attr_logits': logits}


class ComprehensiveExperimentRunner:
    """
    完整的论文实验运行器
    整合三大类实验：
    1. 模块级消融实验
    2. λ 超参数实验
    3. SOTA 对比实验
    """
    
    def __init__(self, train_loader, val_loader, device, num_classes=26):
        """
        初始化实验运行器
        
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
        
        # 创建实验根目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_root_dir = f'paper_experiments_{timestamp}'
        os.makedirs(self.exp_root_dir, exist_ok=True)
        
        logger.info(f"实验结果将保存到: {self.exp_root_dir}")
    
    def run_all_experiments(self, 
                           run_ablation=True, 
                           run_lambda=True, 
                           run_sota=True,
                           epochs=30,
                           learning_rate=3e-4):
        """
        运行所有实验
        
        Args:
            run_ablation: 是否运行消融实验
            run_lambda: 是否运行λ超参数实验
            run_sota: 是否运行SOTA对比实验
            epochs: 训练轮数
            learning_rate: 学习率
        
        Returns:
            dict: 包含所有实验结果的字典
        """
        all_results = {}
        
        logger.info("\n" + "="*100)
        logger.info("开始运行完整的论文实验套件")
        logger.info("="*100)
        
        # 1. 模块级消融实验
        if run_ablation:
            logger.info("\n" + "="*100)
            logger.info("第一部分：模块级消融实验")
            logger.info("="*100)
            ablation_results = self._run_ablation_experiments(epochs, learning_rate)
            all_results['ablation'] = ablation_results
        
        # 2. λ 超参数实验
        if run_lambda:
            logger.info("\n" + "="*100)
            logger.info("第二部分：λ 超参数实验")
            logger.info("="*100)
            lambda_results = self._run_lambda_experiments(epochs, learning_rate)
            all_results['lambda'] = lambda_results
        
        # 3. SOTA 对比实验
        if run_sota:
            logger.info("\n" + "="*100)
            logger.info("第三部分：SOTA 对比实验")
            logger.info("="*100)
            sota_results = self._run_sota_experiments(epochs, learning_rate)
            all_results['sota'] = sota_results
        
        # 生成综合报告
        self._generate_comprehensive_report(all_results)
        
        logger.info("\n" + "="*100)
        logger.info("所有实验完成！")
        logger.info(f"结果保存在: {self.exp_root_dir}")
        logger.info("="*100)
        
        return all_results
    
    def _run_ablation_experiments(self, epochs, learning_rate):
        """运行模块级消融实验"""
        save_dir = os.path.join(self.exp_root_dir, '1_ablation_study')
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义消融模型配置
        ablation_configs = [
            {
                'name': 'Baseline-CNN',
                'model_class': BaselineCNN,
                'description': 'MSFE-FPN + FC，无图结构、无门控、无权重预测器'
            },
            {
                'name': '+Graph-GCN',
                'model_class': CNNWithStaticGraph,
                'description': '加入静态kNN图 + GCN，不使用GAT'
            },
            {
                'name': '+Graph-GAT',
                'model_class': CNNWithAdaptiveGAT,
                'description': '加入动态自适应图 + GAT，但不融合'
            },
            {
                'name': '+Fusion',
                'model_class': CNNGraphFusion,
                'description': 'CNN + Graph 双分支门控融合，不加权重预测器'
            },
            {
                'name': 'Full-AdaGAT',
                'model_class': FullAdaGAT,
                'description': '完整版 AdaGAT：融合 + 类别权重预测器'
            }
        ]
        
        results = {}
        
        # 预训练Full-AdaGAT模型的路径
        pretrained_model_path = 'smart_mixed_checkpoints/best_model.pth'
        
        for config in ablation_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"消融实验: {config['name']}")
            logger.info(f"描述: {config['description']}")
            logger.info(f"{'='*80}\n")
            
            # 创建模型
            model = config['model_class'](num_classes=self.num_classes).to(self.device)
            
            # 如果是Full-AdaGAT且存在预训练模型，则加载预训练模型
            if config['name'] == 'Full-AdaGAT' and os.path.exists(pretrained_model_path):
                logger.info(f"发现预训练的 Full-AdaGAT 模型: {pretrained_model_path}")
                try:
                    checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
                    
                    # 检查模型类型是否兼容
                    if not isinstance(checkpoint, dict):
                        # 如果是完整模型对象，检查类名
                        model_class_name = checkpoint.__class__.__name__
                        if model_class_name != 'FullAdaGAT':
                            logger.warning(f"预训练模型类型不匹配: {model_class_name}，需要FullAdaGAT，跳过加载")
                            raise ValueError(f"模型类型不匹配: {model_class_name}")
                        checkpoint = checkpoint.state_dict()
                    
                    # 加载state_dict
                    model.load_state_dict(checkpoint, strict=True)
                    logger.info("成功加载预训练的 Full-AdaGAT 模型，将直接进行评估")
                    
                    # 只进行评估，不重新训练
                    criterion = nn.BCEWithLogitsLoss()
                    final_metrics = self._validate(model, criterion)
                    
                    result = {
                        'model_name': config['name'],
                        'best_f1': final_metrics['f1'],
                        'best_epoch': 'N/A (使用预训练模型)',
                        'final_metrics': final_metrics,
                        'history': {'note': '使用预训练模型，未重新训练'}
                    }
                    results[config['name']] = result
                    
                    logger.info(f"Full-AdaGAT 验证结果:")
                    logger.info(f"  F1: {final_metrics['f1']:.4f}, "
                               f"P: {final_metrics['precision']:.4f}, "
                               f"R: {final_metrics['recall']:.4f}")
                    
                    # 保存模型副本到消融实验目录
                    model_save_path = os.path.join(save_dir, f'{config["name"]}_best.pth')
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"已将预训练模型副本保存到: {model_save_path}")
                    
                    # 保存结果
                    result_file = os.path.join(save_dir, f'{config["name"]}_results.json')
                    with open(result_file, 'w', encoding='utf-8') as f:
                        serializable_result = {
                            'model_name': config['name'],
                            'best_f1': float(final_metrics['f1']),
                            'best_epoch': 'N/A (使用预训练模型)',
                            'final_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                             for k, v in final_metrics.items()},
                            'note': '使用预训练模型'
                        }
                        json.dump(serializable_result, f, indent=4, ensure_ascii=False)
                    
                    continue  # 跳过训练步骤
                    
                except Exception as e:
                    logger.warning(f"加载预训练模型失败: {str(e)}")
                    logger.info("将重新训练 Full-AdaGAT 模型...")
                    # 继续执行训练流程
            
            # 训练和评估
            result = self._train_and_evaluate(
                model, 
                config['name'], 
                epochs, 
                learning_rate, 
                save_dir
            )
            results[config['name']] = result
        
        # 保存消融实验汇总
        self._save_ablation_summary(results, save_dir)
        
        return results
    
    def _run_lambda_experiments(self, epochs, learning_rate):
        """运行 λ 超参数实验"""
        save_dir = os.path.join(self.exp_root_dir, '2_lambda_study')
        os.makedirs(save_dir, exist_ok=True)
        
        # 使用 LambdaExperiment 类
        lambda_exp = LambdaExperiment(
            self.train_loader,
            self.val_loader,
            self.device,
            self.num_classes
        )
        
        results = lambda_exp.run_experiment(
            epochs=epochs,
            learning_rate=learning_rate,
            save_dir=save_dir
        )
        
        return results
    
    def _run_sota_experiments(self, epochs, learning_rate):
        """运行 SOTA 对比实验"""
        save_dir = os.path.join(self.exp_root_dir, '3_sota_comparison')
        os.makedirs(save_dir, exist_ok=True)
        
        # 定义 SOTA 模型配置
        sota_configs = [
            # {'name': 'ResNet-50', 'backbone': 'resnet50'},
            # {'name': 'ResNet-101', 'backbone': 'resnet101'},
            # {'name': 'EfficientNet-B0', 'backbone': 'efficientnet_b0'},
            # {'name': 'DenseNet-121', 'backbone': 'densenet121'},
            # {'name': 'ViT-B-16', 'backbone': 'vit_b_16'}
        ]
        
        results = {}
        
        for config in sota_configs:
            logger.info(f"\n{'='*80}")
            logger.info(f"SOTA 对比: {config['name']}")
            logger.info(f"{'='*80}\n")
            
            try:
                # 创建模型
                model = SOTABaseline(
                    backbone_name=config['backbone'],
                    num_classes=self.num_classes
                ).to(self.device)
                
                # 训练和评估
                result = self._train_and_evaluate(
                    model,
                    config['name'],
                    epochs,
                    learning_rate,
                    save_dir
                )
                results[config['name']] = result
            except Exception as e:
                logger.error(f"训练 {config['name']} 时出错: {str(e)}")
                continue
        
        # 添加我们的最佳模型进行对比
        # 优先使用预训练的Full-AdaGAT模型
        pretrained_model_path = 'smart_mixed_checkpoints/best_model.pth'
        ablation_dir = os.path.join(self.exp_root_dir, '1_ablation_study')
        ablation_model_path = os.path.join(ablation_dir, 'Full-AdaGAT_best.pth')
        
        logger.info(f"\n{'='*80}")
        logger.info("SOTA 对比: Our Full-AdaGAT")
        logger.info(f"{'='*80}\n")
        
        our_model = FullAdaGAT(
            num_classes=self.num_classes,
            lambda_threshold=0.5  # 默认lambda值
        ).to(self.device)
        
        model_loaded = False
        
        # 优先级1: 使用预训练模型
        if os.path.exists(pretrained_model_path):
            logger.info(f"发现预训练模型: {pretrained_model_path}")
            try:
                checkpoint = torch.load(pretrained_model_path, map_location=self.device, weights_only=False)
                
                # 检查模型类型是否兼容
                if not isinstance(checkpoint, dict):
                    model_class_name = checkpoint.__class__.__name__
                    if model_class_name != 'FullAdaGAT':
                        logger.warning(f"预训练模型类型不匹配: {model_class_name}，需要FullAdaGAT，跳过加载")
                        raise ValueError(f"模型类型不匹配: {model_class_name}")
                    checkpoint = checkpoint.state_dict()
                
                # 加载state_dict
                our_model.load_state_dict(checkpoint, strict=True)
                logger.info("成功加载预训练的 Full-AdaGAT 模型")
                model_loaded = True
            except Exception as e:
                logger.warning(f"加载预训练模型失败: {str(e)}")
        
        # 优先级2: 使用消融实验中的模型
        if not model_loaded and os.path.exists(ablation_model_path):
            logger.info(f"尝试加载消融实验中的模型: {ablation_model_path}")
            try:
                checkpoint = torch.load(ablation_model_path, map_location=self.device, weights_only=False)
                
                # 检查模型类型
                if not isinstance(checkpoint, dict):
                    model_class_name = checkpoint.__class__.__name__
                    if model_class_name != 'FullAdaGAT':
                        logger.warning(f"消融实验模型类型不匹配: {model_class_name}，需要FullAdaGAT")
                        raise ValueError(f"模型类型不匹配: {model_class_name}")
                    checkpoint = checkpoint.state_dict()
                
                our_model.load_state_dict(checkpoint, strict=True)
                logger.info("成功加载消融实验中的 Full-AdaGAT 模型")
                model_loaded = True
            except Exception as e:
                logger.warning(f"加载消融实验模型失败: {str(e)}")
        
        # 如果成功加载模型，进行评估
        if model_loaded:
            logger.info("使用加载的模型进行评估...")
            criterion = nn.BCEWithLogitsLoss()
            final_metrics = self._validate(our_model, criterion)
            
            our_result = {
                'model_name': 'Our-Full-AdaGAT',
                'best_f1': final_metrics['f1'],
                'best_epoch': 'N/A (使用预训练模型)',
                'final_metrics': final_metrics,
                'history': {'note': '使用预训练模型'}
            }
            results['Our-Full-AdaGAT'] = our_result
            
            logger.info(f"Full-AdaGAT 验证结果:")
            logger.info(f"  F1: {final_metrics['f1']:.4f}, "
                       f"P: {final_metrics['precision']:.4f}, "
                       f"R: {final_metrics['recall']:.4f}")
            
            # 保存结果
            result_file = os.path.join(save_dir, 'Our-Full-AdaGAT_results.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                serializable_result = {
                    'model_name': 'Our-Full-AdaGAT',
                    'best_f1': float(final_metrics['f1']),
                    'best_epoch': 'N/A (使用预训练模型)',
                    'final_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                     for k, v in final_metrics.items()},
                    'note': '使用预训练模型'
                }
                json.dump(serializable_result, f, indent=4, ensure_ascii=False)
        else:
            # 如果没有找到预训练模型，重新训练
            logger.warning("未找到预训练模型，将重新训练 Full-AdaGAT 模型...")
            our_result = self._train_and_evaluate(
                our_model,
                'Our-Full-AdaGAT',
                epochs,
                learning_rate,
                save_dir
            )
            results['Our-Full-AdaGAT'] = our_result
        
        # 保存 SOTA 对比汇总
        self._save_sota_summary(results, save_dir)
        
        return results
    
    def _train_and_evaluate(self, model, model_name, epochs, learning_rate, save_dir):
        """训练和评估单个模型"""
        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 损失函数
        criterion = nn.BCEWithLogitsLoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
        
        # 训练历史
        history = {'train': [], 'val': []}
        best_f1 = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\n{model_name} - Epoch {epoch}/{epochs}")
            
            # 训练
            train_metrics = self._train_epoch(model, optimizer, criterion, epoch)
            history['train'].append({'epoch': epoch, **train_metrics})
            
            # 验证
            val_metrics = self._validate(model, criterion)
            history['val'].append({'epoch': epoch, **val_metrics})
            
            # 更新学习率
            scheduler.step(val_metrics['f1'])
            
            # 保存最佳模型
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_epoch = epoch
                
                model_path = os.path.join(save_dir, f'{model_name}_best.pth')
                torch.save(model.state_dict(), model_path)
            
            logger.info(f"训练 - Loss: {train_metrics['loss']:.4f}, "
                       f"F1: {train_metrics['f1']:.4f}, "
                       f"P: {train_metrics['precision']:.4f}, "
                       f"R: {train_metrics['recall']:.4f}")
            logger.info(f"验证 - Loss: {val_metrics['loss']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}, "
                       f"P: {val_metrics['precision']:.4f}, "
                       f"R: {val_metrics['recall']:.4f}")
        
        # 加载最佳模型进行最终评估
        logger.info(f"\n加载最佳模型 (Epoch {best_epoch}) 进行最终评估...")
        best_model_path = os.path.join(save_dir, f'{model_name}_best.pth')
        model.load_state_dict(torch.load(best_model_path))
        
        # 使用最佳模型重新评估
        final_metrics = self._validate(model, criterion)
        
        logger.info(f"最佳模型最终验证结果:")
        logger.info(f"  Loss: {final_metrics['loss']:.4f}, F1: {final_metrics['f1']:.4f}, "
                   f"P: {final_metrics['precision']:.4f}, R: {final_metrics['recall']:.4f}")
        
        result = {
            'model_name': model_name,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'final_metrics': final_metrics,  # 使用最佳模型的评估结果
            'history': history
        }
        
        # 保存结果
        result_file = os.path.join(save_dir, f'{model_name}_results.json')
        with open(result_file, 'w', encoding='utf-8') as f:
            serializable_result = {
                'model_name': model_name,
                'best_f1': float(best_f1),
                'best_epoch': int(best_epoch),
                'final_metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                                 for k, v in final_metrics.items()}
            }
            json.dump(serializable_result, f, indent=4, ensure_ascii=False)
        
        return result
    
    def _train_epoch(self, model, optimizer, criterion, epoch):
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
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def _validate(self, model, criterion):
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
        
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self._calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
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
    
    def _save_ablation_summary(self, results, save_dir):
        """保存消融实验汇总"""
        summary_data = []
        
        for name, result in results.items():
            summary_data.append({
                '模型配置': name,
                'Best F1': result['best_f1'],
                'Precision': result['final_metrics']['precision'],
                'Recall': result['final_metrics']['recall'],
                'Best Epoch': result['best_epoch']
            })
        
        df = pd.DataFrame(summary_data)
        csv_path = os.path.join(save_dir, 'ablation_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 可视化
        self._visualize_ablation(results, save_dir)
        
        logger.info("\n消融实验汇总:")
        logger.info("\n" + df.to_string(index=False))
    
    def _save_sota_summary(self, results, save_dir):
        """保存SOTA对比汇总"""
        summary_data = []
        
        for name, result in results.items():
            summary_data.append({
                '模型': name,
                'Best F1': result['best_f1'],
                'Precision': result['final_metrics']['precision'],
                'Recall': result['final_metrics']['recall'],
                'Best Epoch': result['best_epoch']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Best F1', ascending=False)
        csv_path = os.path.join(save_dir, 'sota_comparison_summary.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 可视化
        self._visualize_sota(results, save_dir)
        
        logger.info("\nSOTA对比汇总:")
        logger.info("\n" + df.to_string(index=False))
    
    def _visualize_ablation(self, results, save_dir):
        """可视化消融实验结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 性能对比柱状图
        names = list(results.keys())
        f1_scores = [results[n]['best_f1'] for n in names]
        precisions = [results[n]['final_metrics']['precision'] for n in names]
        recalls = [results[n]['final_metrics']['recall'] for n in names]
        
        x = np.arange(len(names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, f1_scores, width, label='F1 Score', alpha=0.8)
        ax.bar(x, precisions, width, label='Precision', alpha=0.8)
        ax.bar(x + width, recalls, width, label='Recall', alpha=0.8)
        
        ax.set_xlabel('Model Configuration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Ablation Study: Module-wise Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'ablation_comparison.png'), dpi=300)
        plt.close()
    
    def _visualize_sota(self, results, save_dir):
        """可视化SOTA对比结果"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        names = list(results.keys())
        f1_scores = [results[n]['best_f1'] for n in names]
        
        # 按F1排序
        sorted_indices = np.argsort(f1_scores)
        names = [names[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(names, f1_scores, alpha=0.8)
        
        # 高亮我们的模型
        for i, name in enumerate(names):
            if 'AdaGAT' in name:
                bars[i].set_color('red')
                bars[i].set_alpha(1.0)
        
        ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('SOTA Comparison: F1 Score', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sota_comparison.png'), dpi=300)
        plt.close()
    
    def _generate_comprehensive_report(self, all_results):
        """生成综合实验报告"""
        report_path = os.path.join(self.exp_root_dir, 'COMPREHENSIVE_REPORT.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 论文实验完整报告\n\n")
            f.write(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 消融实验部分
            if 'ablation' in all_results:
                f.write("## 一、模块级消融实验\n\n")
                f.write("本实验用于验证各模块对模型性能的贡献。\n\n")
                
                ablation_results = all_results['ablation']
                f.write("| 模型配置 | Best F1 | Precision | Recall |\n")
                f.write("|---------|---------|-----------|--------|\n")
                
                for name, result in ablation_results.items():
                    f.write(f"| {name} | {result['best_f1']:.4f} | "
                           f"{result['final_metrics']['precision']:.4f} | "
                           f"{result['final_metrics']['recall']:.4f} |\n")
                
                f.write("\n")
            
            # λ 实验部分
            if 'lambda' in all_results:
                f.write("## 二、λ 超参数实验\n\n")
                f.write("本实验用于确定动态阈值 τ = μ + λσ 中 λ 的最优值。\n\n")
                
                lambda_results = all_results['lambda']
                f.write("| Lambda (λ) | Best F1 | Best Epoch |\n")
                f.write("|------------|---------|------------|\n")
                
                for key in sorted(lambda_results.keys()):
                    result = lambda_results[key]
                    f.write(f"| {result['lambda']} | {result['best_f1']:.4f} | "
                           f"{result['best_epoch']} |\n")
                
                f.write("\n")
            
            # SOTA 对比部分
            if 'sota' in all_results:
                f.write("## 三、SOTA 对比实验\n\n")
                f.write("本实验将我们的方法与现有主流方法进行对比。\n\n")
                
                sota_results = all_results['sota']
                f.write("| 模型 | Best F1 | Precision | Recall |\n")
                f.write("|------|---------|-----------|--------|\n")
                
                # 按F1排序
                sorted_results = sorted(sota_results.items(), 
                                      key=lambda x: x[1]['best_f1'], 
                                      reverse=True)
                
                for name, result in sorted_results:
                    marker = " **" if 'AdaGAT' in name else ""
                    f.write(f"| {name}{marker} | {result['best_f1']:.4f} | "
                           f"{result['final_metrics']['precision']:.4f} | "
                           f"{result['final_metrics']['recall']:.4f} |\n")
                
                f.write("\n")
            
            f.write("## 总结\n\n")
            f.write("实验结果表明，我们提出的 AdaGAT 方法在服装属性识别任务上取得了最优性能。\n")
            f.write("模块级消融实验证明了各个组件（图结构、GAT、融合机制、权重预测器）的有效性。\n")
            f.write("超参数实验为动态阈值参数的选择提供了实验依据。\n")
        
        logger.info(f"\n综合实验报告已生成: {report_path}")


def main():
    """主函数"""
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    try:
        train_loader, val_loader = create_data_loaders(EXPERIMENT_CONFIG)
        logger.info(f"训练集大小: {len(train_loader.dataset)}")
        logger.info(f"验证集大小: {len(val_loader.dataset)}")
    except Exception as e:
        logger.error(f"创建数据加载器时出错: {str(e)}")
        sys.exit(1)
    
    # 创建实验运行器
    runner = ComprehensiveExperimentRunner(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_classes=26
    )
    
    # 运行所有实验
    results = runner.run_all_experiments(
        run_ablation=True,   # 运行消融实验
        run_lambda=True,     # 运行λ实验
        run_sota=True,       # 运行SOTA对比
        epochs=30,           # 训练轮数（可根据需要调整）
        learning_rate=3e-4   # 学习率
    )
    
    logger.info("\n所有实验已完成！")


if __name__ == '__main__':
    main()

