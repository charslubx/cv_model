import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import logging
from datetime import datetime
import json
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from collections import defaultdict
from itertools import combinations
import torch.nn.functional as F

sys.path.append('..')
from base_model import MultiScaleFeatureExtractor, FullModel
from training import DeepFashionDataset, collate_fn

class SingleScaleModel(nn.Module):
    """单尺度特征提取模型"""
    def __init__(self, cnn_type='resnet50', weights='IMAGENET1K_V1', num_classes=26):
        super().__init__()
        
        # 只使用最后一层特征
        self.feature_extractor = MultiScaleFeatureExtractor(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048,
            layers_to_extract=['layer4']  # 只使用最后一层
        )
        
        # 添加分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 将特征图转换为向量
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        final_features = features['final']  # [B, 2048, H, W]
        logits = self.classifier(final_features)  # [B, num_classes]
        return logits

class FeatureAttention(nn.Module):
    """特征注意力模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.fc(out)
        return x * attention

class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(out))
        return x * attention

class AblationModel(nn.Module):
    """改进的消融实验模型"""
    def __init__(self, cnn_type='resnet50', weights='IMAGENET1K_V1', num_classes=26, layers_to_use=None):
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractor(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048,
            layers_to_extract=layers_to_use if layers_to_use else ['layer1', 'layer2', 'layer3', 'layer4']
        )
        
        # 特征维度统一
        self.channel_adjusters = nn.ModuleDict({
            'layer1': nn.Conv2d(256, 512, 1),
            'layer2': nn.Conv2d(512, 512, 1),
            'layer3': nn.Conv2d(1024, 512, 1),
            'layer4': nn.Conv2d(2048, 512, 1)
        })
        
        # 添加注意力模块
        self.channel_attention = nn.ModuleDict({
            layer: FeatureAttention(512) for layer in ['layer1', 'layer2', 'layer3', 'layer4']
        })
        self.spatial_attention = nn.ModuleDict({
            layer: SpatialAttention() for layer in ['layer1', 'layer2', 'layer3', 'layer4']
        })
        
        # 自适应特征融合
        self.fusion_weights = nn.Parameter(torch.ones(len(layers_to_use) if layers_to_use else 4))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 保存要使用的层
        self.layers_to_use = layers_to_use if layers_to_use else ['layer1', 'layer2', 'layer3', 'layer4']
        
    def _process_features(self, features, layer_name):
        """处理单层特征"""
        # 应用通道调整
        features = self.channel_adjusters[layer_name](features)
        
        # 应用注意力机制
        features = self.channel_attention[layer_name](features)
        features = self.spatial_attention[layer_name](features)
        
        return features
        
    def forward(self, x):
        # 首先通过初始层
        x = self.feature_extractor.initial_layers(x)
        
        # 获取每一层的特征
        features_dict = {}
        current_feat = x
        
        # 按顺序通过所有必要的层
        for name, layer in self.feature_extractor.feature_layers.items():
            current_feat = layer(current_feat)
            if name in self.layers_to_use:
                features_dict[name] = current_feat
            
            # 如果已经处理完最后一个需要的层，就可以停止
            if name == self.layers_to_use[-1]:
                break
        
        if not features_dict:
            raise ValueError("没有可用的特征层")
        
        # 处理并对齐特征
        processed_features = []
        target_size = None
        
        # 确定目标特征图大小（使用最小的特征图大小）
        for name in self.layers_to_use:
            feat = features_dict[name]
            if target_size is None or feat.shape[2] < target_size[0]:
                target_size = (feat.shape[2], feat.shape[3])
        
        # 处理每层特征
        for name in self.layers_to_use:
            feat = features_dict[name]
            # 调整特征图大小
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            # 处理特征
            processed_feat = self._process_features(feat, name)
            processed_features.append(processed_feat)
            
        # 自适应特征融合
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        fused_features = sum(w * f for w, f in zip(fusion_weights, processed_features))
        
        # 分类
        logits = self.classifier(fused_features)
        
        # 确保输出格式一致
        return {'attr_logits': logits}

def evaluate_model(model, data_loader, device, attr_names):
    """全面评估模型性能"""
    model.eval()
    all_predictions = []
    all_labels = []
    total_time = 0
    num_samples = 0
    per_attr_correct = defaultdict(int)
    per_attr_total = defaultdict(int)
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['attr_labels'].to(device)
            
            # 测量推理时间
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_time += (end_time - start_time)
            
            # 处理不同类型的模型输出
            if isinstance(outputs, dict):
                logits = outputs['attr_logits']
            elif isinstance(outputs, torch.Tensor):
                logits = outputs
            else:
                raise ValueError(f"不支持的模型输出类型: {type(outputs)}")
            
            # 使用sigmoid获取预测概率
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            # 收集预测和标签
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # 统计每个属性的正确预测数
            for i, attr_name in enumerate(attr_names):
                correct = (predictions[:, i] == labels[:, i]).sum().item()
                total = labels.size(0)
                per_attr_correct[attr_name] += correct
                per_attr_total[attr_name] += total
            
            num_samples += images.size(0)
    
    # 合并所有预测和标签
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 计算每个属性的指标
    per_attr_metrics = {}
    overall_precision = []
    overall_recall = []
    overall_f1 = []
    
    for i, attr_name in enumerate(attr_names):
        attr_pred = all_predictions[:, i]
        attr_true = all_labels[:, i]
        
        # 计算该属性的指标
        precision = precision_score(attr_true, attr_pred, zero_division=0)
        recall = recall_score(attr_true, attr_pred, zero_division=0)
        f1 = f1_score(attr_true, attr_pred, zero_division=0)
        
        per_attr_metrics[attr_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        overall_precision.append(precision)
        overall_recall.append(recall)
        overall_f1.append(f1)
    
    # 计算总体指标
    accuracy = np.mean(all_predictions == all_labels)
    avg_precision = np.mean(overall_precision)
    avg_recall = np.mean(overall_recall)
    avg_f1 = np.mean(overall_f1)
    
    # 计算每个属性的准确率
    per_attr_accuracy = {
        attr: correct/total 
        for attr, (correct, total) in zip(
            attr_names, 
            zip(per_attr_correct.values(), per_attr_total.values())
        )
    }
    
    # 计算混淆矩阵
    conf_matrices = {}
    for i, attr_name in enumerate(attr_names):
        conf_matrices[attr_name] = confusion_matrix(
            all_labels[:, i],
            all_predictions[:, i]
        ).tolist()
    
    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'inference_time': total_time / num_samples,
        'per_attr_accuracy': per_attr_accuracy,
        'per_attr_metrics': per_attr_metrics,
        'confusion_matrices': conf_matrices
    }

def run_ablation_experiments(
    data_root='/home/cv_model/deepfashion',
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """运行消融实验"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 定义文件路径
    category_root = os.path.join(data_root, "Category and Attribute Prediction Benchmark")
    anno_dir = os.path.join(category_root, "Anno_fine")
    img_dir = os.path.join(category_root, "Img", "img")
    
    # 验证集文件
    val_img_list = os.path.join(anno_dir, "val.txt")
    val_attr_file = os.path.join(anno_dir, "val_attr.txt")
    
    # 创建验证数据集
    val_dataset = DeepFashionDataset(
        img_list_file=val_img_list,
        attr_file=val_attr_file,
        image_dir=img_dir,
        transform=transform
    )
    
    # 创建验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 获取属性名称
    attr_names = val_dataset.get_attr_names()
    
    # 所有可用的特征层
    all_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    # 创建不同的层组合
    layer_combinations = []
    # 单层实验
    layer_combinations.extend([[layer] for layer in all_layers])
    # 累积层实验
    for i in range(2, len(all_layers) + 1):
        layer_combinations.append(all_layers[:i])
    # 特定组合实验
    additional_combinations = [
        ['layer1', 'layer4'],
        ['layer2', 'layer4'],
        ['layer3', 'layer4'],
        ['layer1', 'layer2', 'layer4'],
        ['layer2', 'layer3', 'layer4']
    ]
    layer_combinations.extend([comb for comb in additional_combinations if comb not in layer_combinations])
    
    results = {}
    
    # 对每个层组合进行实验
    for layers in layer_combinations:
        model_name = f"Ablation_{'_'.join(l.replace('layer', 'L') for l in layers)}"
        logging.info(f"开始评估层组合 {model_name}...")
        
        try:
            # 创建模型
            model = AblationModel(layers_to_use=layers)
            model = model.to(device)
            
            # 评估模型
            metrics = evaluate_model(model, val_loader, device, attr_names)
            results[model_name] = metrics
            
            logging.info(f"完成层组合 {model_name} 的评估")
            logging.info(f"准确率: {metrics['accuracy']:.4f}, F1分数: {metrics['f1_score']:.4f}")
        except Exception as e:
            logging.error(f"评估层组合 {model_name} 时出错: {str(e)}")
            continue
    
    return results

def run_scale_comparison_experiment(
    data_root='/home/cv_model/deepfashion',
    batch_size=32,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # 设置日志
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f'scale_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 定义文件路径
    category_root = os.path.join(data_root, "Category and Attribute Prediction Benchmark")
    anno_dir = os.path.join(category_root, "Anno_fine")
    img_dir = os.path.join(category_root, "Img", "img")
    
    # 训练集文件
    train_img_list = os.path.join(anno_dir, "train.txt")
    train_attr_file = os.path.join(anno_dir, "train_attr.txt")
    
    # 验证集文件
    val_img_list = os.path.join(anno_dir, "val.txt")
    val_attr_file = os.path.join(anno_dir, "val_attr.txt")
    
    # 检查必要文件是否存在
    required_files = [
        train_img_list, train_attr_file,
        val_img_list, val_attr_file
    ]
    for file_path in required_files:
        if not os.path.exists(file_path):
            logging.error(f"找不到必要的文件: {file_path}")
            raise FileNotFoundError(f"找不到必要的文件: {file_path}")
    
    # 加载数据集
    try:
        train_dataset = DeepFashionDataset(
            img_list_file=train_img_list,
            attr_file=train_attr_file,
            image_dir=img_dir,
            transform=transform
        )
        
        val_dataset = DeepFashionDataset(
            img_list_file=val_img_list,
            attr_file=val_attr_file,
            image_dir=img_dir,
            transform=transform
        )
    except Exception as e:
        logging.error(f"加载数据集失败: {str(e)}")
        raise
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 获取属性名称
    attr_names = train_dataset.get_attr_names()
    
    # 创建模型
    multi_scale_model = FullModel()
    single_scale_model = SingleScaleModel()
    
    # 检查模型文件
    model_path = '/home/cv_model/checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        logging.error(f"找不到模型文件: {model_path}")
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    # 加载训练好的模型
    try:
        state_dict = torch.load(model_path, weights_only=True)
        multi_scale_model.load_state_dict(state_dict)
        multi_scale_model = multi_scale_model.to(device)
        logging.info("成功加载多尺度模型")
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise
    
    # 将单尺度模型移到设备
    single_scale_model = single_scale_model.to(device)
    
    # 评估基准模型
    results = {}
    
    # 单尺度模型
    logging.info("开始评估单尺度模型...")
    single_scale_metrics = evaluate_model(single_scale_model, val_loader, device, attr_names)
    results['SingleScale'] = single_scale_metrics
    logging.info(f"单尺度模型评估完成")
    
    # 多尺度模型
    logging.info("开始评估多尺度模型...")
    multi_scale_metrics = evaluate_model(multi_scale_model, val_loader, device, attr_names)
    results['MultiScale'] = multi_scale_metrics
    logging.info(f"多尺度模型评估完成")
    
    # 运行消融实验
    logging.info("开始运行消融实验...")
    ablation_results = run_ablation_experiments(data_root, batch_size, device)
    results.update(ablation_results)
    logging.info("消融实验完成")
    
    # 保存实验结果
    results_dir = 'experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 'comprehensive_comparison_results.json')
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"结果已保存到: {results_path}")
    except Exception as e:
        logging.error(f"保存结果失败: {str(e)}")
        raise
    
    return results

if __name__ == '__main__':
    run_scale_comparison_experiment() 