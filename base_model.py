import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv
from torchvision import models
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义数据集路径
DEEPFASHION_ROOT = "/home/cv_model/DeepFashion"
CATEGORY_ROOT = os.path.join(DEEPFASHION_ROOT, "Category and Attribute Prediction Benchmark")
CATEGORY_ANNO_DIR = os.path.join(CATEGORY_ROOT, "Anno_fine")
CATEGORY_ANNO_COARSE_DIR = os.path.join(CATEGORY_ROOT, "Anno_coarse")

# 标注文件路径
ATTR_CLOTH_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_attr_cloth.txt")
CATEGORY_CLOTH_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_category_cloth.txt")
CATEGORY_IMG_FILE = os.path.join(CATEGORY_ANNO_COARSE_DIR, "list_category_img.txt")
ATTR_IMG_FILE = os.path.join(CATEGORY_ANNO_DIR, "list_attr_img.txt")
BBOX_FILE = os.path.join(CATEGORY_ANNO_COARSE_DIR, "list_bbox.txt")
EVAL_PARTITION_FILE = os.path.join(CATEGORY_ROOT, "Eval", "list_eval_partition.txt")


def read_attr_cloth_file(file_path):
    """读取属性定义文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是属性数量
            num_attrs = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是属性定义
            attrs = []
            for line in lines[2:]:
                parts = line.strip().split()
                attr_name = parts[0]
                attr_type = int(parts[1])  # 1-纹理, 2-面料, 3-形状, 4-部件, 5-风格, 6-合身度
                attrs.append({
                    'name': attr_name,
                    'type': attr_type
                })
            return attrs
    except Exception as e:
        print(f"读取属性定义文件失败: {str(e)}")
        return []


def read_category_cloth_file(file_path):
    """读取类别定义文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是类别数量
            num_categories = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是类别定义
            categories = []
            for line in lines[2:]:
                parts = line.strip().split()
                category_name = parts[0]
                category_type = int(parts[1])  # 1-上衣, 2-下装, 3-连衣裙
                categories.append({
                    'name': category_name,
                    'type': category_type
                })
            return categories
    except Exception as e:
        print(f"读取类别定义文件失败: {str(e)}")
        return []


def read_attr_img_file(file_path):
    """读取图片属性标注文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是图片数量
            num_images = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是图片属性标注
            img_attrs = {}
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                attrs = [int(x) for x in parts[1:]]  # 1表示有该属性，-1表示没有
                img_attrs[img_name] = attrs
            return img_attrs
    except Exception as e:
        print(f"读取图片属性标注文件失败: {str(e)}")
        return {}


def read_eval_partition_file(file_path):
    """读取评估分区文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是图片数量
            num_images = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是评估分区信息
            eval_partitions = {}
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                status = parts[1]  # train/val/test
                eval_partitions[img_name] = status
            return eval_partitions
    except Exception as e:
        print(f"读取评估分区文件失败: {str(e)}")
        return {}


def read_bbox_file(file_path):
    """读取边界框标注文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是图片数量
            num_images = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是边界框信息
            bboxes = {}
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                bbox = [int(x) for x in parts[1:]]  # [x1, y1, x2, y2]
                bboxes[img_name] = bbox
            return bboxes
    except Exception as e:
        print(f"读取边界框标注文件失败: {str(e)}")
        return {}


def read_category_img_file(file_path):
    """读取类别图片标注文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 第一行是图片数量
            num_images = int(lines[0].strip())
            # 第二行是表头
            header = lines[1].strip().split()
            # 剩余行是类别标注
            img_categories = {}
            for line in lines[2:]:
                parts = line.strip().split()
                img_name = parts[0]
                category_id = int(parts[1])  # 类别ID
                img_categories[img_name] = category_id
            return img_categories
    except Exception as e:
        print(f"读取类别图片标注文件失败: {str(e)}")
        return {}


# 读取标注信息
ATTR_DEFS = read_attr_cloth_file(ATTR_CLOTH_FILE)
CATEGORY_DEFS = read_category_cloth_file(CATEGORY_CLOTH_FILE)
IMG_ATTRS = read_attr_img_file(ATTR_IMG_FILE)
IMG_CATEGORIES = read_category_img_file(CATEGORY_IMG_FILE)
EVAL_PARTITIONS = read_eval_partition_file(EVAL_PARTITION_FILE)
BBOXES = read_bbox_file(BBOX_FILE)


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
        # FPN输出统一为256通道，所以总通道数 = 256 * 提取层数
        return 256 * len(layers)

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
        # fpn_features = [P1, P2, P3, P4]，每个都是256通道
        # P1: (B,256,56,56), P2: (B,256,28,28), P3: (B,256,14,14), P4: (B,256,7,7)
        
        # 构建FPN特征字典，方便索引
        fpn_dict = {
            'layer1': fpn_features[0],  # P1
            'layer2': fpn_features[1],  # P2
            'layer3': fpn_features[2],  # P3
            'layer4': fpn_features[3]   # P4
        }

        # 特征融合：使用FPN增强后的特征
        if len(self.layers_to_extract) > 1:
            # 选择FPN特征而不是原始特征
            selected_fpn_features = [fpn_dict[name] for name in self.layers_to_extract]
            target_size = selected_fpn_features[-1].shape[2:]
            
            # 上采样对齐（FPN特征都是256通道）
            resized_features = [
                F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
                for feat in selected_fpn_features[:-1]
            ] + [selected_fpn_features[-1]]
            
            # 拼接FPN特征：256*3 = 768通道
            fused = torch.cat(resized_features, dim=1)
        else:
            fused = fpn_dict[self.layers_to_extract[0]]

        # 返回多个特征表示
        return {
            'global': F.normalize(self.fuse(fused), p=2, dim=1),  # 全局特征（使用FPN增强）
            'fpn': fpn_features,  # FPN特征（用于其他任务）
            'final': features['layer4']  # 最后一层原始特征（用于分割）
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
    """支持加权邻接矩阵的魔改多头GAT（可修改部分已标注）"""

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
        self.residual = residual  # 新增
        self.use_edge_weights = use_edge_weights

        # 添加维度校验
        assert out_features % heads == 0, f"out_features({out_features}) must be divisible by heads({heads})"
        self.out_features_per_head = out_features // heads  # 修正为每个头的维度

        # 修正线性层维度
        self.W = nn.Linear(in_features, heads * self.out_features_per_head, bias=False)

        # 修正多头融合层
        self.head_fusion = nn.Sequential(
            nn.Linear(heads * self.out_features_per_head, out_features),  # 输入维度修正
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 注意力系数计算（含边权重）
        if use_edge_weights:
            self.edge_encoder = nn.Linear(edge_dim, self.heads)  # 边权重编码
        self.attn_src = nn.Parameter(torch.Tensor(1, self.heads, self.out_features_per_head))
        self.attn_dst = nn.Parameter(torch.Tensor(1, self.heads, self.out_features_per_head))

        # 残差连接
        if residual:
            self.res_fc = nn.Linear(in_features, heads * self.out_features_per_head)

        # 初始化参数
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if use_edge_weights:
            nn.init.xavier_uniform_(self.edge_encoder.weight)

    def forward(self, x: Tensor, adj_matrix: Tensor) -> Tensor:
        num_nodes = x.size(0)

        # 1. 线性变换 + 分头
        h = self.W(x)  # [num_nodes, heads * out_features_per_head]
        h = h.view(num_nodes, self.heads, self.out_features_per_head)  # [N, H, D_h]

        # 2. 计算注意力分数（含边权重）
        attn_src = torch.sum(h * self.attn_src, dim=-1)  # [N, H]
        attn_dst = torch.sum(h * self.attn_dst, dim=-1)  # [N, H]
        attn = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)  # [N, N, H]

        # 边权重编码（如果启用）
        if self.use_edge_weights:
            edge_weights = self.edge_encoder(adj_matrix.unsqueeze(-1))  # [N, N, H]
            attn = attn + edge_weights

        # 3. 掩码处理（仅保留邻接矩阵中的边）
        mask = (adj_matrix > 0).unsqueeze(-1)  # [N, N, 1]
        attn = attn.masked_fill(~mask, float('-inf'))

        # 4. 注意力归一化
        attn = F.leaky_relu(attn, self.alpha)  # [N, N, H]
        attn = F.softmax(attn, dim=1)  # 按目标节点归一化
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # 5. 特征聚合（加权求和）
        h_out = torch.einsum('ijh,jhd->ihd', attn, h)  # [N, H, D_h]
        h_out = h_out.reshape(num_nodes, self.heads * self.out_features_per_head)  # [N, H*D_h]

        # 6. 残差连接
        if self.res_fc is not None:
            res = self.res_fc(x)
            h_out = h_out + res

        # 修改后的多头融合方式
        h_out = self.head_fusion(h_out)  # 新增多头交互层

        return h_out


class StackedGAT(nn.Module):
    """堆叠多层魔改GAT（可修改部分已标注）"""

    def __init__(self,
                 in_features: int,
                 hidden_dims: list = [1024, 512, 256],  # 确保每层维度是heads的整数倍
                 heads: int = 12,
                 dropout: float = 0.3,
                 edge_dim: int = 1,
                 residual: bool = True):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_features] + hidden_dims

        # 添加维度校验
        for i, h_dim in enumerate(hidden_dims):
            assert h_dim % heads == 0, f"hidden_dim[{i}]({h_dim}) must be divisible by heads({heads})"

        # 构建GAT层
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(
                WeightedMultiHeadGAT(
                    in_features=in_dim,
                    out_features=out_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    residual=residual
                )
            )

    def forward(self, x: Tensor, adj_matrix: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, adj_matrix)
        return x


class GCNClassifier(nn.Module):
    """GCN分类器（可修改部分已标注）"""

    def __init__(self,
                 in_features: int,
                 num_classes: int,  # 可修改：类别数
                 hidden_dims: list = [128]  # 可修改：隐藏层维度
                 ):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_features] + hidden_dims + [num_classes]
        for i in range(len(dims) - 1):
            self.layers.append(
                GCNConv(dims[i], dims[i + 1])
            )
            if i != len(dims) - 2:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(dims[i + 1]))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)
            else:
                x = layer(x)
        return torch.sigmoid(x)  # 多标签分类用Sigmoid


class FullModel(nn.Module):
    """完整的多任务服装分析模型"""

    def __init__(self,
                 num_classes: int = 26,
                 cnn_type: str = 'resnet50',
                 weights: str = 'IMAGENET1K_V1',
                 enable_segmentation: bool = True,
                 enable_textile_classification: bool = True,
                 num_fabric_classes: int = 20,  # fabric类别数
                 num_fiber_classes: int = 32,   # fiber类别数
                 gat_dims: list = [1024, 512],
                 gat_heads: int = 4,
                 gat_dropout: float = 0.2):
        super().__init__()
        self.num_classes = num_classes
        self.enable_segmentation = enable_segmentation
        self.enable_textile_classification = enable_textile_classification
        self.num_fabric_classes = num_fabric_classes
        self.num_fiber_classes = num_fiber_classes

        # 特征提取器 - 增强特征提取能力
        self.feature_extractor = MultiScaleFeatureExtractor(
            cnn_type=cnn_type,
            weights=weights,
            output_dims=2048,
            layers_to_extract=['layer2', 'layer3', 'layer4']
        )

        # 特征增强模块（适中dropout）
        self.feature_enhancer = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.15)  # 适中的dropout
        )

        # GAT特征增强（适中dropout）
        self.gat = StackedGAT(
            in_features=2048,
            hidden_dims=[2048],  # 简化为单层，保持维度一致
            heads=16,  # 增加注意力头数
            dropout=0.2,  # 适中的dropout
            edge_dim=1,
            residual=True
        )

        # GCN分类器（适中dropout）
        self.gcn = GCNClassifier(
            in_features=2048,  # 匹配GAT输出
            num_classes=num_classes,
            hidden_dims=[1024]  # 简化为单层
        )

        # 简化的属性分类头（适中dropout）
        self.attr_head = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.25),  # 适中的dropout
            nn.Linear(1024, num_classes)
        )

        # 分割分支（如果启用）
        if enable_segmentation:
            self.segmentation_branch = SegmentationBranch(
                in_channels=2048,
                num_classes=1
            )

        # 简化的特征融合（适中dropout）
        self.fusion_gate = nn.Sequential(
            nn.Linear(2048 * 2, 256),  # 两个2048维特征
            nn.ReLU(),
            nn.Dropout(0.15),  # 适中dropout
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        # 类别权重预测器（适中dropout）
        self.class_weight_predictor = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # 适中dropout
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
        # 纹理分类头（如果启用）- 使用更深层和专门化的架构
        if enable_textile_classification:
            # 纺织品特征适配层 - 简化架构避免梯度问题
            self.textile_adapter = nn.Sequential(
                nn.Linear(2048, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # Fabric分类头 - 简化架构
            self.fabric_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_fabric_classes)
            )
            
            # Fiber分类头 - 简化架构
            self.fiber_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_fiber_classes)
            )
            
            # 统一纹理分类头（用于混合分类）- 简化架构
            self.textile_head = nn.Sequential(
                nn.Linear(512, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, max(num_fabric_classes, num_fiber_classes))
            )
            
            # 初始化纹理分类头的权重
            self._init_textile_heads()

    def _init_textile_heads(self):
        """初始化纺织品分类头的权重，使用渐进式初始化"""
        def init_adapter_weights(m):
            if isinstance(m, nn.Linear):
                # 适配层使用标准Xavier初始化
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        def init_classifier_weights(m):
            if isinstance(m, nn.Linear):
                # 分类器使用更小的初始化
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 分别初始化不同层
        self.textile_adapter.apply(init_adapter_weights)
        self.fabric_head.apply(init_classifier_weights)
        self.fiber_head.apply(init_classifier_weights)
        self.textile_head.apply(init_classifier_weights)

    def forward(self, images, img_names=None):
        """前向传播"""
        # 提取特征
        features = self.feature_extractor(images)

        # 特征稳定性检查
        if torch.isnan(features['global']).any():
            print("Warning: NaN detected in global features")
            features['global'] = torch.nan_to_num(features['global'], nan=0.0)

        # 准备输出字典
        outputs = {}

        # 1. 特征增强（不归一化，保留原始信息）
        global_features = self.feature_enhancer(features['global'])

        # 2. 构建改进的邻接矩阵
        # 先归一化用于计算相似度
        normalized_features = F.normalize(global_features, p=2, dim=1)
        
        # 计算余弦相似度
        sim_matrix = torch.matmul(normalized_features, normalized_features.t())

        # 使用适中的动态阈值（平衡precision和recall）
        mean_sim = sim_matrix.mean()
        std_sim = sim_matrix.std()
        threshold = mean_sim + 0.4 * std_sim  # 适中的阈值

        # 构建稀疏邻接矩阵
        adj_matrix = (sim_matrix > threshold).float()
        adj_matrix = adj_matrix * sim_matrix  # 保留相似度权重

        # 确保邻接矩阵非空且有自环
        adj_matrix = adj_matrix + torch.eye(adj_matrix.size(0), device=adj_matrix.device)
        
        # 添加top-k连接确保最小连通性
        batch_size = adj_matrix.size(0)
        if batch_size > 1:
            k_neighbors = min(5, batch_size - 1)
            topk_values, topk_indices = torch.topk(sim_matrix, k_neighbors + 1, dim=1)
            for i in range(batch_size):
                for j in range(1, k_neighbors + 1):  # 跳过自己
                    neighbor_idx = topk_indices[i, j]
                    adj_matrix[i, neighbor_idx] = topk_values[i, j]

        # 3. GAT特征增强（使用原始特征，不是归一化后的）
        gat_features = self.gat(global_features, adj_matrix)

        # 4. GCN分类
        edge_index = adj_matrix.nonzero().t()
        gcn_logits = self.gcn(gat_features, edge_index)

        # 5. CNN特征分类
        cnn_logits = self.attr_head(global_features)

        # 6. 预测类别权重（适中范围）
        class_weights = self.class_weight_predictor(global_features)
        class_weights = torch.clamp(class_weights, 0.6, 1.4)  # 适中范围

        # 7. 改进的特征融合
        fusion_input = torch.cat([global_features, gat_features], dim=1)
        fusion_weights = self.fusion_gate(fusion_input)

        # 8. 加权融合（先融合再应用权重）
        attr_logits = fusion_weights[:, 0:1] * gcn_logits + fusion_weights[:, 1:2] * cnn_logits
        
        # 应用类别权重（适度影响）
        attr_logits = attr_logits * (0.7 + 0.3 * class_weights)  # 适度应用权重

        outputs['attr_logits'] = attr_logits
        outputs['class_weights'] = class_weights

        # 9. 分割（如果启用）
        if self.enable_segmentation:
            seg_features = features['final']
            # 分割特征稳定性检查
            if torch.isnan(seg_features).any():
                print("Warning: NaN detected in segmentation features")
                seg_features = torch.nan_to_num(seg_features, nan=0.0)

            seg_logits = self.segmentation_branch(seg_features)
            # 限制分割logits范围
            seg_logits = torch.clamp(seg_logits, -10, 10)
            outputs['seg_logits'] = seg_logits

        # 10. 纹理分类（如果启用）- 使用专门的适配层
        if self.enable_textile_classification:
            # 首先通过适配层转换特征
            adapted_features = self.textile_adapter(global_features)
            
            # Fabric分类 - 使用适配后的特征
            fabric_logits = self.fabric_head(adapted_features)
            outputs['fabric_logits'] = fabric_logits
            
            # Fiber分类 - 使用适配后的特征
            fiber_logits = self.fiber_head(adapted_features)
            outputs['fiber_logits'] = fiber_logits
            
            # 统一纹理分类（用于混合训练）
            textile_logits = self.textile_head(adapted_features)
            outputs['textile_logits'] = textile_logits

        return outputs


class FocalLoss(nn.Module):
    """改进的Focal Loss，支持AMP训练"""

    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出的logits
            targets: 目标值（0或1）
        """
        # 使用binary_cross_entropy_with_logits，它会在内部处理sigmoid
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 计算pt
        pt = torch.exp(-bce_loss)
        
        # 计算focal loss
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        # 添加alpha权重（如果指定）
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_weight * focal_loss

        # 根据reduction方式返回
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
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


    def validate(self):
        # 使用EMA模型进行验证
        if self.current_epoch >= self.ema_start:
            eval_model = self.ema.module
        else:
            eval_model = self.model

        eval_model.eval()
        val_loss = 0.0

        # 存储所有预测和标签
        all_preds = []
        all_labels = []

        # 分割相关指标
        seg_iou = 0.0
        seg_count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                attr_labels = batch['attr_labels'].to(self.device)
                img_names = batch['img_name']

                # 获取分割任务的标签
                seg_labels = batch.get('segmentation')
                if seg_labels is not None:
                    seg_labels = seg_labels.to(self.device)

                # 前向传播
                outputs = eval_model(images, img_names)

                # 1. 属性分类评估
                attr_preds = torch.sigmoid(outputs['attr_logits'])  # 使用sigmoid而不是阈值
                all_preds.append(attr_preds.cpu())
                all_labels.append(attr_labels.cpu())

                # 2. 分割评估
                if self.enable_segmentation and seg_labels is not None:
                    seg_preds = (outputs['seg_logits'] > 0).float()
                    intersection = (seg_preds * seg_labels).sum().item()
                    union = (seg_preds + seg_labels).gt(0).sum().item()
                    if union > 0:
                        seg_iou += intersection / union
                        seg_count += 1

        # 合并所有预测和标签
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 计算多个阈值下的指标
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        best_f1 = 0
        best_threshold = 0.5
        metrics = {}

        for threshold in thresholds:
            binary_preds = (all_preds > threshold).float()

            # 计算每个属性的指标
            tp = (binary_preds * all_labels).sum(dim=0)
            fp = (binary_preds * (1 - all_labels)).sum(dim=0)
            fn = ((1 - binary_preds) * all_labels).sum(dim=0)

            # 计算精确率和召回率
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)

            # 计算F1分数
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

            # 计算平均指标
            avg_precision = precision.mean().item()
            avg_recall = recall.mean().item()
            avg_f1 = f1.mean().item()

            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = threshold
                metrics = {
                    'precision': avg_precision,
                    'recall': avg_recall,
                    'f1': avg_f1,
                    'threshold': threshold
                }

        # 计算分割IoU
        seg_avg_iou = seg_iou / seg_count if seg_count > 0 else 0
        metrics['seg_iou'] = seg_avg_iou

        return metrics


    def train(self, epochs: int = 50, save_dir: str = "checkpoints"):
        """训练模型
        Args:
            epochs: 训练轮数
            save_dir: 模型保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 更新scheduler的epochs参数
        self.scheduler.total_steps = epochs * len(self.train_loader)

        best_f1 = 0.0

        for epoch in range(epochs):
            self.current_epoch = epoch + 1

            # 训练一个epoch
            train_metrics = self.train_epoch()

            # 验证
            val_metrics = self.validate()

            # 输出训练信息
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            logger.info("训练指标:")
            for k, v in train_metrics.items():
                logger.info(f"- {k}: {v:.4f}")

            logger.info("\n验证指标:")
            logger.info(f"- precision: {val_metrics['precision']:.4f}")
            logger.info(f"- recall: {val_metrics['recall']:.4f}")
            logger.info(f"- f1: {val_metrics['f1']:.4f}")
            logger.info(f"- best_threshold: {val_metrics['threshold']:.4f}")
            if self.enable_segmentation:
                logger.info(f"- seg_iou: {val_metrics['seg_iou']:.4f}")

            # 早停检查
            current_f1 = val_metrics['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                self.patience_counter = 0
                # 保存最佳模型（完整模型）
                model_path = os.path.join(save_dir, "best_model.pth")
                torch.save(self.model, model_path)
                logger.info(f"保存最佳模型到: {model_path}")
            else:
                self.patience_counter += 1