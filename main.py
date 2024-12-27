import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from skimage import segmentation, color, io
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# 读取图像
image = io.imread('image_1.jpg')  # <---- 自定义图像路径

# 超像素分割（使用 SLIC 算法）
segments = segmentation.slic(image, n_segments=100, compactness=10)  # <---- 自定义超像素数量和紧凑度

# 显示分割后的图像
plt.imshow(segmentation.mark_boundaries(image, segments))
plt.show()

# 获取每个超像素的特征
def get_superpixel_features(image, segments):
    num_segments = np.max(segments) + 1
    features = np.zeros((num_segments, 3))  # 假设我们用 RGB 特征
    for i in range(num_segments):
        region_mask = (segments == i)
        region_pixels = image[region_mask]
        features[i] = np.mean(region_pixels, axis=0)  # <---- 自定义特征提取方法（例如平均颜色）
    return features

# 获取超像素特征
features = get_superpixel_features(image, segments)

# 归一化特征（L2 归一化）
features = torch.tensor(features, dtype=torch.float32)
features = torch.nn.functional.normalize(features, p=2, dim=1)

# 将特征转换为 tensor
features_tensor = torch.tensor(features, dtype=torch.float)

# 基于超像素的空间邻接关系构建图
def build_adjacency_matrix(segments):
    num_segments = np.max(segments) + 1
    adjacency_matrix = np.zeros((num_segments, num_segments))

    # 遍历相邻的超像素区域，构建邻接矩阵
    for i in range(segments.shape[0] - 1):
        for j in range(segments.shape[1] - 1):
            # 获取当前像素所在的超像素
            curr_segment = segments[i, j]
            # 获取右邻居和下邻居的超像素
            right_segment = segments[i, j + 1] if j + 1 < segments.shape[1] else curr_segment
            down_segment = segments[i + 1, j] if i + 1 < segments.shape[0] else curr_segment

            # 在邻接矩阵中标记邻接关系
            adjacency_matrix[curr_segment, right_segment] = 1
            adjacency_matrix[curr_segment, down_segment] = 1

    # 对邻接矩阵进行归一化
    adjacency_matrix = normalize(adjacency_matrix, norm='l1', axis=1)
    return adjacency_matrix

# 获取邻接矩阵
adjacency_matrix = build_adjacency_matrix(segments)

# 将邻接矩阵转换为 tensor
edge_index = torch.tensor(np.array(np.nonzero(adjacency_matrix)), dtype=torch.long)

# 创建数据对象
from torch_geometric.data import Data
data = Data(x=features_tensor, edge_index=edge_index)

# GCN 模型
import torch_geometric.nn as gnn

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = gnn.GCNConv(input_dim, hidden_dim)  # <---- 可调整隐藏层维度
        self.conv2 = gnn.GCNConv(hidden_dim, output_dim)  # <---- 可调整输出层维度
        self.fc = nn.Linear(output_dim, 10)  # 假设有 10 类 <---- 自定义类别数量

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # GCN 层
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # 分类层
        x = self.fc(x)
        return x

# 创建模型
model = GCN(input_dim=3, hidden_dim=64, output_dim=32)  # <---- 可调整输入、隐藏、输出维度

# 并行神经网络与多头注意力机制
class ParallelAttentionNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(ParallelAttentionNN, self).__init__()
        # 并行神经网络
        self.parallel_layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_heads)])  # <---- 可调整并行层数和隐藏层维度
        # 多头注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)  # <---- 可调整注意力头数
        self.fc = nn.Linear(hidden_dim, output_dim)  # <---- 自定义输出维度

    def forward(self, x):
        # 并行处理
        outputs = [layer(x) for layer in self.parallel_layers]
        outputs = torch.stack(outputs, dim=1)

        # 多头注意力机制
        attn_output, _ = self.attention(outputs, outputs, outputs)

        # 聚合并分类
        out = attn_output.mean(dim=1)
        out = self.fc(out)
        return out

# 添加并行注意力层到模型中
model_with_attention = ParallelAttentionNN(input_dim=32, hidden_dim=64, output_dim=10)  # <---- 可调整输入、隐藏、输出维度

# 模拟标签
labels = torch.randint(0, 10, (features.shape[0],))  # <---- 自定义标签数量与类别

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # <---- 自定义损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # <---- 自定义学习率

# 训练过程
def train(model, data, labels):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

# 训练 10 个 epoch
for epoch in range(10):  # <---- 自定义训练 epoch 数量
    loss = train(model, data, labels)
    print(f"Epoch {epoch+1}, Loss: {loss}")

# 测试过程
def test(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
    return pred

# 假设进行测试
predictions = test(model, data)
print(predictions)

# 结果可视化
plt.imshow(image)
plt.title(f'Predicted Label: {predictions[0].item()}')  # <---- 显示预测标签
plt.show()
