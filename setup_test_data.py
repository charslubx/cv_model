import os
import pandas as pd
import numpy as np
from PIL import Image

# 创建目录
os.makedirs("data/images", exist_ok=True)

# 创建测试图片
def create_test_image(path, size=(224, 224)):
    img = Image.new('RGB', size, color='white')
    img.save(path)

# 创建一些测试图片
n_images = 10
image_paths = []
for i in range(n_images):
    path = f"data/images/test_image_{i}.jpg"
    create_test_image(path)
    image_paths.append(f"test_image_{i}.jpg")

# 创建测试标签CSV
labels = np.random.randint(0, 2, size=(n_images, 20))
df = pd.DataFrame(labels, columns=[f"class_{i}" for i in range(20)])
df['image_path'] = image_paths
df.to_csv("data/test_labels.csv", index=False)

print("测试数据创建完成!") 