import torch
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from skimage import segmentation


class Superpixel:
    def __init__(self, image_url):
        response = requests.get(url=image_url)
        image = Image.open(BytesIO(response.content))  # 从响应中加载图像
        # 转换为 numpy 数组（skimage 或其他库处理时需要）
        self.image = np.array(image)

    def superpixel_image(self):
        # 超像素分割（使用 SLIC 算法）
        segments = segmentation.slic(self.image, n_segments=100,
                                     compactness=10)  # <---- 自定义超像素数量和紧凑度
        return segments

    def get_superpixel_features(self, segments):
        # 超像素分割后提取特征
        num_segments = np.max(segments) + 1
        features = np.zeros((num_segments, 3))  # 假设我们用 RGB 特征
        for i in range(num_segments):
            region_mask = (segments == i)
            region_pixels = self.image[region_mask]
            features[i] = np.mean(region_pixels, axis=0)  # <---- 自定义特征提取方法（例如平均颜色）

        # 归一化特征（L2 归一化）
        features = torch.tensor(features, dtype=torch.float32)
        features = torch.nn.functional.normalize(features, p=2, dim=1)

        # 将特征转换为 tensor
        features_tensor = torch.tensor(features, dtype=torch.float)
        return features_tensor
