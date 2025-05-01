import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch_geometric.nn import GCNConv
from torchvision import models, transforms
import pandas as pd
import os

# 定义数据集路径
DEEPFASHION_ROOT = "deepfashion"
DATA_DIR = "data/deepfashion_merged"
ATTR_STATS_FILE = os.path.join(DATA_DIR, "attribute_stats.csv")

class LandmarkBranch(nn.Module):
    """服装关键点检测分支"""
    def __init__(self, in_channels, num_landmarks=8):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # 关键点检测卷积层
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, num_landmarks * 3, 1)  # *3: x, y, visibility
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        
        # 分离坐标和可见性预测
        B, C, H, W = x.shape
        x = x.view(B, self.num_landmarks, 3, H, W)
        
        # 对坐标进行softmax，分别在H和W维度上进行
        coords_x = F.softmax(x[:, :, 0], dim=-1)  # 在W维度上的softmax
        coords_y = F.softmax(x[:, :, 1], dim=-2)  # 在H维度上的softmax
        coords = torch.stack([coords_x, coords_y], dim=2)
        
        vis = torch.sigmoid(x[:, :, 2].mean(dim=[-2, -1]))  # 可见性预测
        
        return coords, vis

class SegmentationBranch(nn.Module):
    """服装分割分支"""
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, num_classes, 1)
        
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class MultiScaleFeatureExtractor(nn.Module):
    """增强的多尺度特征提取器"""
    def __init__(self,
                 cnn_type: str = 'resnet50',
                 weights: str = 'IMAGENET1K_V1',
                 output_dims: int = 2048,
                 layers_to_extract: list = ['layer2', 'layer3', 'layer4']
                 ):
        super().__init__()
        self.cnn_type = cnn_type
        self.output_dims = output_dims
        self.layers_to_extract = layers_to_extract
        
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
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # 多尺度特征融合
        self.fuse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._get_total_channels(cnn_type, layers_to_extract), output_dims),
            nn.BatchNorm1d(output_dims, affine=True),
            nn.ReLU()
        )
        
    def _get_total_channels(self, cnn_type: str, layers: list) -> int:
        channel_map = {
            'resnet50': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048},
            'resnet101': {'layer1': 256, 'layer2': 512, 'layer3': 1024, 'layer4': 2048}
        }
        return sum([channel_map[cnn_type][layer] for layer in layers])
    
    def forward(self, x: Tensor) -> tuple:
        # 初始特征提取
        x = self.initial_layers(x)
        
        # 提取多尺度特征
        features = {}
        current_feat = x
        for name, layer in self.feature_layers.items():
            current_feat = layer(current_feat)
            features[name] = current_feat
        
        # FPN特征增强
        fpn_features = self.fpn(features)
        
        # 特征融合
        if len(self.layers_to_extract) > 1:
            selected_features = [features[name] for name in self.layers_to_extract]
            target_size = selected_features[-1].shape[2:]
            resized_features = [
                F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                for feat in selected_features[:-1]
            ] + [selected_features[-1]]
            fused = torch.cat(resized_features, dim=1)
        else:
            fused = features[self.layers_to_extract[0]]
        
        # 返回多个特征表示
        return {
            'global': F.normalize(self.fuse(fused), p=2, dim=1),  # 全局特征
            'fpn': fpn_features,  # FPN特征
            'final': features['layer4']  # 最后一层特征
        }

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
            
    def forward(self, x):
        features = list(x.values())
        last_inner = self.inner_blocks[-1](features[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        for idx in range(len(features) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](features[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))
            
        return results

class WeightedMultiHeadGAT(nn.Module):
    """支持加权邻接矩阵的多头GAT"""
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 heads: int = 12,
                 edge_dim: int = 1,
                 dropout: float = 0.3,
                 alpha: float = 0.2,
                 residual: bool = True,
                 use_edge_weights: bool = True):
        super().__init__()
        self.heads = heads
        self.alpha = alpha
        self.dropout = dropout
        self.residual = residual
        self.use_edge_weights = use_edge_weights
        
        assert out_features % heads == 0, f"out_features({out_features}) must be divisible by heads({heads})"
        self.out_features_per_head = out_features // heads
        
        self.W = nn.Linear(in_features, heads * self.out_features_per_head, bias=False)
        
        self.head_fusion = nn.Sequential(
            nn.Linear(heads * self.out_features_per_head, out_features),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if use_edge_weights:
            self.edge_encoder = nn.Linear(edge_dim, self.heads)
        self.attn_src = nn.Parameter(torch.Tensor(1, self.heads, self.out_features_per_head))
        self.attn_dst = nn.Parameter(torch.Tensor(1, self.heads, self.out_features_per_head))
        
        if residual:
            self.res_fc = nn.Linear(in_features, heads * self.out_features_per_head)
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if self.use_edge_weights:
            nn.init.xavier_uniform_(self.edge_encoder.weight)
            
    def forward(self, x: Tensor, adj_matrix: Tensor) -> Tensor:
        num_nodes = x.size(0)
        
        # 线性变换 + 分头
        h = self.W(x)
        h = h.view(num_nodes, self.heads, self.out_features_per_head)
        
        # 计算注意力分数
        attn_src = torch.sum(h * self.attn_src, dim=-1)
        attn_dst = torch.sum(h * self.attn_dst, dim=-1)
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        
        # 边权重编码
        if self.use_edge_weights:
            edge_weights = self.edge_encoder(adj_matrix.unsqueeze(-1))
            attn = attn + edge_weights
            
        # 掩码处理
        mask = (adj_matrix > 0).unsqueeze(-1)
        attn = attn.masked_fill(~mask, float('-inf'))
        
        # 注意力归一化
        attn = F.leaky_relu(attn, self.alpha)
        attn = F.softmax(attn, dim=1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # 特征聚合
        h_out = torch.einsum('ijh,jhd->ihd', attn, h)
        h_out = h_out.reshape(num_nodes, self.heads * self.out_features_per_head)
        
        # 残差连接
        if self.residual and self.res_fc is not None:
            res = self.res_fc(x)
            h_out = h_out + res
            
        # 多头融合
        h_out = self.head_fusion(h_out)
        
        return h_out

class FullModel(nn.Module):
    """增强的端到端模型"""
    def __init__(self,
                 cnn_feat_dim: int = 2048,
                 gat_dims: list = None,  # 修改为可选参数
                 num_classes: int = None,
                 gat_heads: int = 12,
                 gat_dropout: float = 0.3,
                 enable_landmarks: bool = True,
                 enable_segmentation: bool = True):
        super().__init__()
        
        # 如果没有指定GAT维度，则根据CNN特征维度自动设置
        # 确保所有维度都能被注意力头数整除
        if gat_dims is None:
            gat_dims = [cnn_feat_dim, 1032, 504]  # 1032和504都能被12整除
            
        # 读取属性信息
        if num_classes is None:
            attr_stats = pd.read_csv(ATTR_STATS_FILE)
            num_classes = len(attr_stats)
            
        self.enable_landmarks = enable_landmarks
        self.enable_segmentation = enable_segmentation
        
        # CNN特征提取器
        self.cnn = MultiScaleFeatureExtractor(
            cnn_type='resnet50',
            weights='IMAGENET1K_V1',
            output_dims=cnn_feat_dim,
            layers_to_extract=['layer2', 'layer3', 'layer4']
        )
        
        # GAT特征增强
        self.gat = nn.ModuleList([
            WeightedMultiHeadGAT(
                in_features=gat_dims[i],
                out_features=gat_dims[i+1],
                heads=gat_heads,
                dropout=gat_dropout
            )
            for i in range(len(gat_dims)-1)
        ])
        
        # 属性分类器
        self.classifier = nn.Sequential(
            nn.Linear(gat_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 关键点检测分支
        if enable_landmarks:
            self.landmark_branch = LandmarkBranch(
                in_channels=2048,  # ResNet最后一层的通道数
                num_landmarks=8  # 全身服装的关键点数
            )
            
        # 分割分支
        if enable_segmentation:
            self.segmentation_branch = SegmentationBranch(
                in_channels=2048,
                num_classes=1  # 二值分割
            )
            
        # 特征归一化
        self.gat_norm = nn.LayerNorm(gat_dims[-1])
        
    def forward(self, x_img: Tensor) -> dict:
        # 1. CNN特征提取
        features = self.cnn(x_img)
        
        # 2. 构建加权邻接矩阵
        global_features = features['global']
        sim_matrix = F.cosine_similarity(
            global_features.unsqueeze(1),
            global_features.unsqueeze(0),
            dim=2
        )
        adj_matrix = (sim_matrix + 1) / 2
        
        # 3. GAT特征增强
        x = global_features
        for gat_layer in self.gat:
            x = gat_layer(x, adj_matrix)
        x = self.gat_norm(x)
        x = F.leaky_relu(x, 0.2)
        
        # 4. 属性分类
        attr_logits = self.classifier(x)
        
        # 准备输出
        outputs = {
            'attr_logits': attr_logits,
            'features': features
        }
        
        # 5. 关键点检测（如果启用）
        if self.enable_landmarks:
            landmark_coords, landmark_vis = self.landmark_branch(features['final'])
            outputs.update({
                'landmark_coords': landmark_coords,
                'landmark_vis': landmark_vis
            })
            
        # 6. 分割（如果启用）
        if self.enable_segmentation:
            seg_logits = self.segmentation_branch(features['final'])
            outputs['seg_logits'] = seg_logits
            
        return outputs

class FocalLoss(nn.Module):
    """改进的Focal Loss"""
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class MultiTaskLoss(nn.Module):
    """多任务损失"""
    def __init__(self,
                 num_tasks=3,
                 reduction='mean',
                 device=None):
        super().__init__()
        self.num_tasks = num_tasks
        self.reduction = reduction
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化任务权重并移动到正确的设备
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, device=self.device))
        
    def forward(self, losses):
        # 确保losses在正确的设备上
        if not isinstance(losses, torch.Tensor):
            losses = torch.stack(losses).to(self.device)
        elif losses.device != self.device:
            losses = losses.to(self.device)
            
        # 动态权重
        weights = torch.exp(-self.log_vars)
        
        # 加权损失
        weighted_losses = weights * losses + 0.5 * self.log_vars
        
        if self.reduction == 'mean':
            return weighted_losses.mean()
        elif self.reduction == 'sum':
            return weighted_losses.sum()
        return weighted_losses

# 测试用例
if __name__ == "__main__":
    # 初始化完整模型
    model = FullModel(
        enable_landmarks=True,
        enable_segmentation=True
    )
    
    # 测试单张图像
    x = torch.randn(1, 3, 224, 224)
    outputs = model(x)
    
    print("输出:")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v.shape}")
