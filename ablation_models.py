"""
消融实验模型变体
用于论文第四章的模块级消融实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torchvision import models
from typing import Optional


class MultiScaleFeatureExtractorBase(nn.Module):
    """基础多尺度特征提取器（用于所有消融变体）"""
    
    def __init__(self,
                 cnn_type: str = 'resnet50',
                 weights: str = 'IMAGENET1K_V1',
                 output_dims: int = 2048):
        super().__init__()
        self.cnn_type = cnn_type
        self.output_dims = output_dims
        
        # 加载预训练主干网络
        if weights == 'IMAGENET1K_V1':
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        elif weights == 'DEFAULT':
            weights = models.ResNet50_Weights.DEFAULT
        elif weights is None:
            weights = None
            
        backbone = models.__dict__[cnn_type](weights=weights)
        
        # 提取所有必要层
        self.initial_layers = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        
        self.feature_layers = nn.ModuleDict({
            'layer1': backbone.layer1,
            'layer2': backbone.layer2,
            'layer3': backbone.layer3,
            'layer4': backbone.layer4
        })
        
        # 特征融合层
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, output_dims),
            nn.BatchNorm1d(output_dims, affine=True),
            nn.ReLU()
        )
    
    def forward(self, x: Tensor) -> dict:
        """提取多尺度特征"""
        # 初始特征提取
        x = self.initial_layers(x)
        
        # 提取多尺度特征
        features = {}
        current_feat = x
        for name, layer in self.feature_layers.items():
            current_feat = layer(current_feat)
            features[name] = current_feat
        
        # 全局特征
        global_features = self.fuse(features['layer4'])
        
        return {
            'global': global_features,
            'layer4': features['layer4'],
            'all_features': features
        }


class BaselineCNN(nn.Module):
    """
    Baseline-CNN：MSFE-FPN + FC
    无图结构、无门控、无权重预测器
    这是最基础的对比模型
    """
    
    def __init__(self, num_classes=26, cnn_type='resnet50', weights='IMAGENET1K_V1'):
        super().__init__()
        self.num_classes = num_classes
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractorBase(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048
        )
        
        # 简单的分类头
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, images):
        features = self.feature_extractor(images)
        global_features = features['global']
        
        # 直接分类
        attr_logits = self.classifier(global_features)
        
        return {'attr_logits': attr_logits}


class GCNClassifier(nn.Module):
    """简单的GCN分类器"""
    
    def __init__(self, in_features: int, hidden_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)
        self.dropout = dropout
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class CNNWithStaticGraph(nn.Module):
    """
    + Graph-GCN（静态图）
    使用简单的kNN图 + GCN，不使用GAT，不使用门控
    这个变体用于证明"有图结构但没有注意力/融合时"的水平
    """
    
    def __init__(self, num_classes=26, cnn_type='resnet50', weights='IMAGENET1K_V1', k_neighbors=5):
        super().__init__()
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractorBase(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048
        )
        
        # GCN分类器
        self.gcn = GCNClassifier(
            in_features=2048,
            hidden_features=1024,
            num_classes=num_classes,
            dropout=0.3
        )
    
    def forward(self, images):
        features = self.feature_extractor(images)
        global_features = features['global']
        
        # 构建简单的kNN图
        sim_matrix = F.cosine_similarity(
            global_features.unsqueeze(1),
            global_features.unsqueeze(0),
            dim=2
        )
        
        # kNN图构建
        batch_size = sim_matrix.size(0)
        k = min(self.k_neighbors, batch_size - 1)
        topk_values, topk_indices = torch.topk(sim_matrix, k + 1, dim=1)  # +1 因为包含自己
        
        # 构建邻接矩阵
        adj_matrix = torch.zeros_like(sim_matrix)
        for i in range(batch_size):
            adj_matrix[i, topk_indices[i]] = topk_values[i]
        
        # 确保邻接矩阵非空
        if adj_matrix.sum() == 0:
            adj_matrix = torch.eye(batch_size, device=adj_matrix.device)
        
        # GCN分类
        edge_index = adj_matrix.nonzero().t()
        gcn_logits = self.gcn(global_features, edge_index)
        
        return {'attr_logits': gcn_logits}


class WeightedMultiHeadGAT(nn.Module):
    """支持加权邻接矩阵的多头GAT（从base_model.py复用）"""
    
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 heads: int = 8,
                 edge_dim: int = 1,
                 dropout: float = 0.3,
                 alpha: float = 0.2,
                 residual: bool = True,
                 use_edge_weights: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout
        self.alpha = alpha
        self.residual = residual
        self.use_edge_weights = use_edge_weights
        
        self.out_features_per_head = out_features // heads
        assert self.out_features_per_head * heads == out_features, "out_features必须能被heads整除"
        
        # 线性变换
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # 注意力参数
        self.attn_src = nn.Parameter(torch.zeros(1, heads, self.out_features_per_head))
        self.attn_dst = nn.Parameter(torch.zeros(1, heads, self.out_features_per_head))
        
        # 边权重编码器
        if use_edge_weights:
            self.edge_encoder = nn.Linear(edge_dim, heads)
        
        # 残差连接
        if residual and in_features != out_features:
            self.res_fc = nn.Linear(in_features, out_features, bias=False)
        else:
            self.res_fc = None
        
        # 多头融合层
        self.head_fusion = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU()
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if self.res_fc is not None:
            nn.init.xavier_uniform_(self.res_fc.weight)
        if self.use_edge_weights:
            nn.init.xavier_uniform_(self.edge_encoder.weight)
    
    def forward(self, x: Tensor, adj_matrix: Tensor) -> Tensor:
        num_nodes = x.size(0)
        
        # 1. 线性变换 + 分头
        h = self.W(x)
        h = h.view(num_nodes, self.heads, self.out_features_per_head)
        
        # 2. 计算注意力分数
        attn_src = torch.sum(h * self.attn_src, dim=-1)
        attn_dst = torch.sum(h * self.attn_dst, dim=-1)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        
        # 边权重编码
        if self.use_edge_weights:
            edge_weights = self.edge_encoder(adj_matrix.unsqueeze(-1))
            attn = attn + edge_weights
        
        # 3. 掩码处理
        mask = (adj_matrix > 0).unsqueeze(-1)
        attn = attn.masked_fill(~mask, float('-inf'))
        
        # 4. 注意力归一化
        attn = F.leaky_relu(attn, self.alpha)
        attn = F.softmax(attn, dim=1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # 5. 特征聚合
        h_out = torch.einsum('ijh,jhd->ihd', attn, h)
        h_out = h_out.reshape(num_nodes, self.heads * self.out_features_per_head)
        
        # 6. 残差连接
        if self.res_fc is not None:
            res = self.res_fc(x)
            h_out = h_out + res
        
        # 多头融合
        h_out = self.head_fusion(h_out)
        
        return h_out


class CNNWithAdaptiveGAT(nn.Module):
    """
    + Graph-GAT（AdaGAT分支，但不融合）
    开启加权多头GAT + GCN，使用动态自适应阈值图
    但只用图分支的logits做预测（等价于门控权重全给图分支）
    用来证明"图注意力 + GCN"相对于纯CNN/纯GCN的价值
    """
    
    def __init__(self, num_classes=26, cnn_type='resnet50', weights='IMAGENET1K_V1', 
                 lambda_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_threshold = lambda_threshold
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractorBase(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048
        )
        
        # 特征增强器
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # GAT（加权多头注意力）
        self.gat = WeightedMultiHeadGAT(
            in_features=2048,
            out_features=2048,
            heads=8,
            dropout=0.3,
            use_edge_weights=True
        )
        
        # GCN分类器
        self.gcn = GCNClassifier(
            in_features=2048,
            hidden_features=1024,
            num_classes=num_classes,
            dropout=0.3
        )
    
    def forward(self, images):
        features = self.feature_extractor(images)
        global_features = features['global']
        
        # 特征增强
        global_features = self.feature_enhancer(global_features)
        global_features = F.normalize(global_features, p=2, dim=1)
        
        # 构建动态自适应阈值图
        sim_matrix = F.cosine_similarity(
            global_features.unsqueeze(1),
            global_features.unsqueeze(0),
            dim=2
        )
        
        # 使用动态阈值（τ = μ + λσ）
        mean_sim = sim_matrix.mean()
        std_sim = sim_matrix.std()
        threshold = mean_sim + self.lambda_threshold * std_sim
        
        # 构建稀疏邻接矩阵
        adj_matrix = (sim_matrix > threshold).float()
        adj_matrix = adj_matrix * sim_matrix  # 保留相似度权重
        
        # 确保邻接矩阵非空
        if adj_matrix.sum() == 0:
            adj_matrix = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        
        # GAT特征增强
        gat_features = self.gat(global_features, adj_matrix)
        gat_features = F.normalize(gat_features, p=2, dim=1)
        
        # GCN分类（只用图分支）
        edge_index = adj_matrix.nonzero().t()
        gcn_logits = self.gcn(gat_features, edge_index)
        
        return {'attr_logits': gcn_logits}


class CNNGraphFusion(nn.Module):
    """
    + Fusion（CNN + Graph 双分支门控融合）
    CNN logits + 图分支 logits 进入 fusion gate
    不加类别权重预测器，损失用普通BCE或Focal
    用来单独说明"门控融合"带来的提升
    """
    
    def __init__(self, num_classes=26, cnn_type='resnet50', weights='IMAGENET1K_V1',
                 lambda_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_threshold = lambda_threshold
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractorBase(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048
        )
        
        # 特征增强器
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # GAT
        self.gat = WeightedMultiHeadGAT(
            in_features=2048,
            out_features=2048,
            heads=8,
            dropout=0.3,
            use_edge_weights=True
        )
        
        # GCN分类器（图分支）
        self.gcn = GCNClassifier(
            in_features=2048,
            hidden_features=1024,
            num_classes=num_classes,
            dropout=0.3
        )
        
        # CNN分类器（CNN分支）
        self.attr_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, images):
        features = self.feature_extractor(images)
        global_features = features['global']
        
        # 特征增强
        global_features = self.feature_enhancer(global_features)
        global_features = F.normalize(global_features, p=2, dim=1)
        
        # 构建动态自适应阈值图
        sim_matrix = F.cosine_similarity(
            global_features.unsqueeze(1),
            global_features.unsqueeze(0),
            dim=2
        )
        
        # 使用动态阈值
        mean_sim = sim_matrix.mean()
        std_sim = sim_matrix.std()
        threshold = mean_sim + self.lambda_threshold * std_sim
        
        # 构建稀疏邻接矩阵
        adj_matrix = (sim_matrix > threshold).float()
        adj_matrix = adj_matrix * sim_matrix
        
        if adj_matrix.sum() == 0:
            adj_matrix = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        
        # GAT特征增强
        gat_features = self.gat(global_features, adj_matrix)
        gat_features = F.normalize(gat_features, p=2, dim=1)
        
        # 图分支分类
        edge_index = adj_matrix.nonzero().t()
        gcn_logits = self.gcn(gat_features, edge_index)
        
        # CNN分支分类
        cnn_logits = self.attr_head(global_features)
        
        # 门控融合
        fusion_input = torch.cat([global_features, gat_features], dim=1)
        fusion_weights = self.fusion_gate(fusion_input)
        
        # 加权融合
        attr_logits = fusion_weights[:, 0:1] * gcn_logits + fusion_weights[:, 1:2] * cnn_logits
        
        return {'attr_logits': attr_logits}


class FullAdaGAT(nn.Module):
    """
    + Class-Weight Predictor（完整版 AdaGAT）
    在上一行基础上再加类别权重预测器 + Focal Loss
    这就是论文里要称之为 "Full AdaGAT" 的最终模型
    """
    
    def __init__(self, num_classes=26, cnn_type='resnet50', weights='IMAGENET1K_V1',
                 lambda_threshold=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_threshold = lambda_threshold
        
        # 特征提取器
        self.feature_extractor = MultiScaleFeatureExtractorBase(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048
        )
        
        # 特征增强器
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # GAT
        self.gat = WeightedMultiHeadGAT(
            in_features=2048,
            out_features=2048,
            heads=8,
            dropout=0.3,
            use_edge_weights=True
        )
        
        # GCN分类器
        self.gcn = GCNClassifier(
            in_features=2048,
            hidden_features=1024,
            num_classes=num_classes,
            dropout=0.3
        )
        
        # CNN分类器
        self.attr_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        # 门控融合机制
        self.fusion_gate = nn.Sequential(
            nn.Linear(2048 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
        # 类别权重预测器（用于长尾处理）
        self.class_weight_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, images):
        features = self.feature_extractor(images)
        global_features = features['global']
        
        # 特征增强
        global_features = self.feature_enhancer(global_features)
        global_features = F.normalize(global_features, p=2, dim=1)
        
        # 构建动态自适应阈值图
        sim_matrix = F.cosine_similarity(
            global_features.unsqueeze(1),
            global_features.unsqueeze(0),
            dim=2
        )
        
        # 使用动态阈值
        mean_sim = sim_matrix.mean()
        std_sim = sim_matrix.std()
        threshold = mean_sim + self.lambda_threshold * std_sim
        
        # 构建稀疏邻接矩阵
        adj_matrix = (sim_matrix > threshold).float()
        adj_matrix = adj_matrix * sim_matrix
        
        if adj_matrix.sum() == 0:
            adj_matrix = torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        
        # GAT特征增强
        gat_features = self.gat(global_features, adj_matrix)
        gat_features = F.normalize(gat_features, p=2, dim=1)
        
        # 图分支分类
        edge_index = adj_matrix.nonzero().t()
        gcn_logits = self.gcn(gat_features, edge_index)
        
        # CNN分支分类
        cnn_logits = self.attr_head(global_features)
        
        # 预测类别权重
        class_weights = self.class_weight_predictor(global_features)
        class_weights = torch.clamp(class_weights, 0.1, 1.0)
        
        # 门控融合
        fusion_input = torch.cat([global_features, gat_features], dim=1)
        fusion_weights = self.fusion_gate(fusion_input)
        
        # 加权融合
        attr_logits = fusion_weights[:, 0:1] * gcn_logits + fusion_weights[:, 1:2] * cnn_logits
        attr_logits = attr_logits * class_weights  # 应用类别权重
        
        # 输出稳定性检查
        attr_logits = torch.clamp(attr_logits, -10, 10)
        
        return {
            'attr_logits': attr_logits,
            'class_weights': class_weights
        }

