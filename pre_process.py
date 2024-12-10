import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


def load_and_preprocess_image(img_path, img_size=224):
    # 加载图片
    img = image.load_img(img_path, target_size=(img_size, img_size))

    # 转换为数组并扩展批量维度
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 使其成为 (1, img_size, img_size, 3)

    # 归一化到 [0, 1]
    img_array = img_array / 255.0  # 或者你可以根据需要使用其他归一化方法

    return img_array
