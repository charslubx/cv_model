import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from experiment_config import EXPERIMENT_CONFIG, MetricsCalculator, ExperimentLogger
import logging
import os
from tqdm import tqdm

class ExperimentRunner:
    def __init__(self, config, models, train_loader, val_loader, device):
        self.config = config
        self.models = models
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = ExperimentLogger('clothing_attribute_recognition', list(models.keys()))
        self.metrics_calculator = MetricsCalculator()
        
    def train_epoch(self, model, optimizer, criterion, model_name, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Training {model_name} Epoch {epoch}')
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
            
        # 计算epoch级别的指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.metrics_calculator.calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self, model, criterion, model_name):
        """验证模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f'Validating {model_name}'):
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
        metrics = self.metrics_calculator.calculate_metrics(all_preds, all_targets)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def run_experiment(self):
        """运行完整实验"""
        results = {}
        
        for model_name, model in self.models.items():
            logging.info(f"\n开始训练模型: {model_name}")
            model = model.to(self.device)
            
            # 优化器和损失函数
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            criterion = nn.BCEWithLogitsLoss()
            
            # 学习率调度器
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                patience=self.config['training']['scheduler_patience'],
                factor=0.1
            )
            
            best_f1 = 0
            patience_counter = 0
            
            for epoch in range(self.config['training']['epochs']):
                # 训练
                train_metrics = self.train_epoch(model, optimizer, criterion, model_name, epoch)
                self.logger.log_metrics(model_name, 'train', epoch, train_metrics)
                
                # 验证
                val_metrics = self.validate(model, criterion, model_name)
                self.logger.log_metrics(model_name, 'val', epoch, val_metrics)
                
                # 更新学习率
                scheduler.step(val_metrics['f1'])
                
                # 早停检查
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['training']['early_stopping_patience']:
                        logging.info(f"Early stopping triggered for {model_name}")
                        break
                
                logging.info(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
                logging.info(f"Train Metrics: {train_metrics}")
                logging.info(f"Val Metrics: {val_metrics}")
            
            results[model_name] = {'best_f1': best_f1}
        
        # 保存实验结果
        self.logger.save_results('results')
        return results
    
    def run_ablation_study(self):
        """运行消融实验"""
        ablation_results = {}
        base_model = self.models['gcn_gat']
        
        # 特征融合消融
        if self.config['ablation']['feature_fusion']:
            model_no_fusion = base_model.copy()
            model_no_fusion.fusion_gate = None
            ablation_results['no_fusion'] = self.run_single_model(model_no_fusion, 'no_fusion')
        
        # 多尺度特征消融
        if self.config['ablation']['multi_scale']:
            model_single_scale = base_model.copy()
            model_single_scale.feature_extractor.layers_to_extract = ['layer4']
            ablation_results['single_scale'] = self.run_single_model(model_single_scale, 'single_scale')
        
        # 注意力机制消融
        if self.config['ablation']['attention_mechanism']:
            model_no_attention = base_model.copy()
            model_no_attention.gat = None
            ablation_results['no_attention'] = self.run_single_model(model_no_attention, 'no_attention')
        
        # 图构建消融
        if self.config['ablation']['graph_construction']:
            model_no_graph = base_model.copy()
            model_no_graph.gcn = None
            ablation_results['no_graph'] = self.run_single_model(model_no_graph, 'no_graph')
        
        return ablation_results
    
    def run_single_model(self, model, model_name):
        """运行单个模型实验"""
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_metrics = None
        for epoch in range(self.config['training']['epochs']):
            train_metrics = self.train_epoch(model, optimizer, criterion, model_name, epoch)
            val_metrics = self.validate(model, criterion, model_name)
            
            if best_metrics is None or val_metrics['f1'] > best_metrics['f1']:
                best_metrics = val_metrics
        
        return best_metrics 