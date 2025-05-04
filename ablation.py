import os
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from evaluation import ModelEvaluator
import torch.nn as nn

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureProcessor:
    """特征处理器"""
    
    @staticmethod
    def process_features(features: torch.Tensor, target_dim: int = 512) -> torch.Tensor:
        """
        处理特征张量，统一维度
        
        Args:
            features: 输入特征张量
            target_dim: 目标特征维度
            
        Returns:
            处理后的特征张量
        """
        try:
            # 确保输入是有效的张量
            if not isinstance(features, torch.Tensor):
                raise ValueError("输入必须是torch.Tensor类型")
            
            # 获取原始维度
            original_shape = features.shape
            
            # 处理FPN特征（如果是列表）
            if isinstance(features, (list, tuple)):
                # 对FPN特征进行平均
                features = torch.stack(features, dim=0).mean(dim=0)
            
            # 处理不同维度的输入
            if features.dim() == 2:  # [B, C]
                features = features.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            elif features.dim() == 3:  # [B, C, S] 或 [B, S, C]
                if features.size(1) > features.size(-1):  # 判断哪个维度是通道维
                    features = features.unsqueeze(-1)  # [B, C, S, 1]
                else:
                    features = features.transpose(1, 2).unsqueeze(-1)  # [B, C, S, 1]
            elif features.dim() == 4:  # [B, C, H, W]
                pass  # 已经是正确的格式
            else:
                raise ValueError(f"不支持的输入维度: {features.dim()}")
            
            # 统一空间维度
            features = F.adaptive_avg_pool2d(features, (1, 1))  # [B, C, 1, 1]
            features = features.squeeze(-1).squeeze(-1)  # [B, C]
            
            # 调整通道数
            if features.size(-1) != target_dim:
                # 使用1x1卷积调整通道数
                conv = nn.Conv1d(features.size(-1), target_dim, 1).to(features.device)
                features = features.unsqueeze(-1)  # [B, C, 1]
                features = conv(features)  # [B, target_dim, 1]
                features = features.squeeze(-1)  # [B, target_dim]
            
            # 返回处理后的特征
            return features
            
        except Exception as e:
            logger.error(f"特征处理失败: {str(e)}")
            logger.error(f"输入特征形状: {original_shape}")
            raise

class AblationStudy:
    """消融实验分析器"""
    
    def __init__(self, model_class, data_loader, device, attr_names):
        """
        初始化消融实验分析器
        
        Args:
            model_class: 模型类
            data_loader: 数据加载器
            device: 计算设备
            attr_names: 属性名称列表
        """
        self.model_class = model_class
        self.data_loader = data_loader
        self.device = device
        self.attr_names = attr_names
        self.feature_processor = FeatureProcessor()
        
        # 可用的特征层
        self.available_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        
        # 验证初始化参数
        self._validate_initialization()
        
    def _validate_initialization(self):
        """验证初始化参数"""
        if self.data_loader is None:
            raise ValueError("data_loader 不能为 None")
        if self.attr_names is None or len(self.attr_names) == 0:
            raise ValueError("attr_names 不能为空")
    
    def run_ablation(self, save_dir: str = "ablation_results") -> Dict:
        """
        执行消融实验
        
        Args:
            save_dir: 结果保存目录
            
        Returns:
            dict: 包含消融实验结果的字典
        """
        os.makedirs(save_dir, exist_ok=True)
        results = {}
        
        # 生成所有可能的层组合
        layer_combinations = []
        
        # 1. 单层实验
        layer_combinations.extend([[layer] for layer in self.available_layers])
        
        # 2. 两层组合
        for i in range(len(self.available_layers)):
            for j in range(i + 1, len(self.available_layers)):
                layer_combinations.append([self.available_layers[i], self.available_layers[j]])
        
        # 3. 三层组合
        for i in range(len(self.available_layers)):
            for j in range(i + 1, len(self.available_layers)):
                for k in range(j + 1, len(self.available_layers)):
                    layer_combinations.append([self.available_layers[i], self.available_layers[j], self.available_layers[k]])
        
        # 4. 全层组合
        layer_combinations.append(self.available_layers)
        
        # 执行每个组合的实验
        for layers in layer_combinations:
            config_name = f"Ablation_{'_'.join(l.replace('layer', 'L') for l in layers)}"
            logger.info(f"开始评估层组合 {config_name}...")
            
            try:
                # 创建模型实例
                model = self.model_class(layers_to_use=layers).to(self.device)
                
                # 获取特征提取器
                feature_extractor = None
                for module in model.modules():
                    if isinstance(module, MultiScaleFeatureExtractor):
                        feature_extractor = module
                        break
                
                if feature_extractor is None:
                    raise ValueError("找不到特征提取器")
                
                # 存储中间特征
                features_dict = {}
                
                def feature_hook(name):
                    def hook(module, input, output):
                        # 处理特征并存储
                        processed = self.feature_processor.process_features(output)
                        if name not in features_dict:
                            features_dict[name] = []
                        features_dict[name].append(processed)
                    return hook
                
                # 添加特征处理钩子
                hooks = []
                for layer in layers:
                    if layer in feature_extractor.feature_layers:
                        hook = feature_extractor.feature_layers[layer].register_forward_hook(feature_hook(layer))
                        hooks.append(hook)
                
                # 修改模型的forward方法
                original_forward = model.forward
                
                def new_forward(x, *args, **kwargs):
                    # 清空特征字典
                    features_dict.clear()
                    
                    # 运行原始forward
                    output = original_forward(x, *args, **kwargs)
                    
                    # 如果没有特征，说明钩子没有被触发
                    if not features_dict:
                        raise ValueError("没有可用的特征层")
                    
                    # 融合特征
                    all_features = []
                    for layer in layers:
                        if layer in features_dict:
                            all_features.extend(features_dict[layer])
                    
                    # 如果有多个特征，取平均
                    if len(all_features) > 1:
                        target_size = all_features[-1].shape[2:]
                        resized_features = [
                            F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                            for feat in all_features[:-1]
                        ] + [all_features[-1]]
                        fused = torch.cat(resized_features, dim=1)
                    else:
                        fused = all_features[0]
                    
                    # 返回处理后的特征
                    return {'attr_logits': output['attr_logits'] if isinstance(output, dict) else output}
                
                # 替换forward方法
                model.forward = new_forward.__get__(model)
                
                # 评估模型
                evaluator = ModelEvaluator(model, self.data_loader, self.device, self.attr_names)
                metrics = evaluator.evaluate(os.path.join(save_dir, config_name))
                results[config_name] = metrics
                
                # 移除钩子
                for hook in hooks:
                    hook.remove()
                
                # 恢复原始forward方法
                model.forward = original_forward
                    
            except Exception as e:
                logger.error(f"评估层组合 {config_name} 时出错: {str(e)}")
                continue
        
        # 保存和可视化结果
        if results:
            self._save_results(results, save_dir)
            self._visualize_results(results, save_dir)
        else:
            logger.error("没有成功的实验结果可以保存")
        
        return results
    
    def _save_results(self, results: Dict, save_dir: str):
        """保存实验结果"""
        # 准备数据
        data = []
        for config_name, metrics in results.items():
            row = {
                'configuration': config_name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'inference_time': metrics['inference_time']
            }
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(save_dir, 'ablation_results.csv'), index=False)
        
        # 打印结果摘要
        logger.info("\n=== 消融实验结果摘要 ===")
        logger.info("\n最佳配置:")
        best_config = df.loc[df['f1'].idxmax()]
        logger.info(f"配置: {best_config['configuration']}")
        logger.info(f"F1分数: {best_config['f1']:.4f}")
        logger.info(f"准确率: {best_config['accuracy']:.4f}")
        logger.info(f"推理时间: {best_config['inference_time']:.4f}秒")
    
    def _visualize_results(self, results: Dict, save_dir: str):
        """可视化实验结果"""
        # 1. 性能对比图
        plt.figure(figsize=(15, 6))
        
        # 准备数据
        configs = list(results.keys())
        accuracies = [results[c]['accuracy'] for c in configs]
        f1_scores = [results[c]['f1'] for c in configs]
        inference_times = [results[c]['inference_time'] for c in configs]
        
        # 绘制性能指标
        x = np.arange(len(configs))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='准确率')
        plt.bar(x + width/2, f1_scores, width, label='F1分数')
        
        plt.xlabel('模型配置')
        plt.ylabel('分数')
        plt.title('不同特征层组合的性能对比')
        plt.xticks(x, configs, rotation=45, ha='right')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'performance_comparison.png'))
        plt.close()
        
        # 2. 推理时间对比图
        plt.figure(figsize=(12, 6))
        plt.bar(configs, inference_times)
        plt.xlabel('模型配置')
        plt.ylabel('推理时间 (秒)')
        plt.title('不同特征层组合的推理时间对比')
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_comparison.png'))
        plt.close()
        
        # 3. 热力图
        metrics_data = {
            config: {
                'accuracy': results[config]['accuracy'],
                'precision': results[config]['precision'],
                'recall': results[config]['recall'],
                'f1': results[config]['f1'],
                'inference_time': results[config]['inference_time']
            }
            for config in configs
        }
        
        metrics_df = pd.DataFrame(metrics_data).T
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='YlOrRd')
        plt.title('各配置的性能指标热力图')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_heatmap.png'))
        plt.close()

    def _process_features(self, features, layer_name):
        # 根据层名选择对应的处理器
        adjuster = self.channel_adjusters[layer_name]
        attention_c = self.channel_attention[layer_name]
        attention_s = self.spatial_attention[layer_name]
        
        # 依次应用通道调整和注意力机制
        features = adjuster(features)
        features = attention_c(features)
        features = attention_s(features)
        
        # 自适应权重
        fusion_weights = F.softmax(self.fusion_weights, dim=0)
        # 加权融合
        fused_features = sum(w * f for w, f in zip(fusion_weights, features))
        
        return fused_features

class FeatureAttention(nn.Module):
    def forward(self, x):
        # 结合最大池化和平均池化的特征
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        out = torch.cat([avg_out, max_out], dim=1)
        # 通过FC层生成注意力权重
        attention = self.fc(out)
        return x * attention 

channel_map = {
    'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
    'resnet101': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
} 

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self,
                 cnn_type: str = 'resnet50',
                 weights: str = 'IMAGENET1K_V1',
                 output_dims: int = 2048,
                 layers_to_extract: list = None):
        super().__init__()
        self.cnn_type = cnn_type
        self.output_dims = output_dims
        # 确保layers_to_extract有效
        self.layers_to_extract = layers_to_extract or ['layer2', 'layer3', 'layer4']
        
        # 验证层的有效性
        valid_layers = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer in self.layers_to_extract:
            if layer not in valid_layers:
                raise ValueError(f"Invalid layer: {layer}")

        # 初始化特征提取器
        self.initial_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 定义特征层
        self.feature_layers = nn.ModuleDict({
            layer: self._get_layer(layer) for layer in self.layers_to_extract
        })

    def _get_layer(self, layer_name):
        if self.cnn_type == 'resnet50':
            if layer_name == 'layer1':
                return self.initial_layers
            elif layer_name == 'layer2':
                return nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            elif layer_name == 'layer3':
                return nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
            elif layer_name == 'layer4':
                return nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
        elif self.cnn_type == 'resnet101':
            if layer_name == 'layer1':
                return self.initial_layers
            elif layer_name == 'layer2':
                return nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
            elif layer_name == 'layer3':
                return nn.Sequential(
                    nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
            elif layer_name == 'layer4':
                return nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                )
        else:
            raise ValueError(f"Unsupported cnn_type: {self.cnn_type}")

    def forward(self, x: torch.Tensor) -> dict:
        # 初始特征提取
        x = self.initial_layers(x)
        
        # 提取多尺度特征
        features = {}
        current_feat = x
        for name, layer in self.feature_layers.items():
            current_feat = layer(current_feat)
            features[name] = current_feat
        
        # 验证特征是否存在
        selected_features = []
        for layer_name in self.layers_to_extract:
            if layer_name not in features:
                raise ValueError(f"Feature layer {layer_name} not found")
            selected_features.append(features[layer_name])
        
        if not selected_features:
            raise ValueError("No valid feature layers selected")
        
        return selected_features

    def fuse_features(self, features: list) -> torch.Tensor:
        """改进的特征融合机制"""
        if len(features) == 0:
            raise ValueError("No features to fuse")
        elif len(features) == 1:
            return features[0]
        
        # 获取目标尺寸（使用最小的特征图）
        target_size = None
        for feat in features:
            if target_size is None or feat.shape[2] < target_size[0]:
                target_size = (feat.shape[2], feat.shape[3])
        
        # 特征对齐和融合
        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                   mode='bilinear', align_corners=False)
            aligned_features.append(feat)
        
        # 通道注意力融合
        fused = torch.cat(aligned_features, dim=1)
        return self.channel_attention(fused)

    def process_features(self, x: torch.Tensor) -> dict:
        """处理特征并返回结果"""
        try:
            # 特征提取
            features = self.extract_features(x)
            
            # 特征融合
            fused = self.fuse_features(features)
            
            # 返回结果
            return {
                'global': F.normalize(self.fuse(fused), p=2, dim=1),
                'features': features,
                'fused': fused
            }
        except Exception as e:
            logger.error(f"Error processing features: {str(e)}")
            raise 