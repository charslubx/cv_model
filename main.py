import tensorflow as tf
from keras.src import activations

from pre_process import load_and_preprocess_image


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, activation, patch_size, embed_dim, stride=16):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride
        self.activation = activation

        # 使用深度可分离卷积
        # 可考虑替换为标准卷积，或增加卷积核数目以提升模型表达能力
        self.conv1 = tf.keras.layers.SeparableConv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.stride,
            padding='same',
            activation=None
        )

        self.act = self.activation

    def call(self, x):
        x = self.conv1(x)
        x = self.act(x)
        return x


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout
        )
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embed_dim)
        ])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, mask=None):
        # Multi-Head Attention
        attn_output = self.attention(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # 残差连接

        # Feedforward Network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差连接

        return out2


class VisionTransformer(tf.keras.Model):
    def __init__(self, activation, img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12, ff_dim=2048, num_classes=1000):
        super(VisionTransformer, self).__init__()

        # Patch嵌入层
        self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, stride=patch_size, activation=activation)

        # 位置编码层，使用可学习的位置编码
        num_patches = (img_size // patch_size) ** 2
        self.position_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=embed_dim)

        # Transformer 编码器层
        self.encoder_layers = [TransformerEncoderLayer(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]

        # 分类头
        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        # 确保输入是 float32 类型
        x = tf.cast(x, tf.float32)

        # 通过 PatchEmbed 层将输入图像转换为 patch 嵌入
        x = self.patch_embed(x)

        # 扁平化图像，准备传入 Transformer
        x = tf.reshape(x, (x.shape[0], -1, x.shape[-1]))  # (batch_size, num_patches, embed_dim)

        # 生成位置编码，并将其加到 patch 嵌入上
        num_patches = x.shape[1]
        position_encoding = self.position_embedding(tf.range(num_patches))  # (num_patches,)
        position_encoding = tf.expand_dims(position_encoding, 0)  # 扩展到 batch_size 维度
        x = x + position_encoding  # 位置编码与 patch 嵌入相加

        # 经过每个 Transformer 编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # 对输出进行池化（例如取均值池化）
        # 可根据需求替换为更复杂的池化策略（例如最大池化或加权池化）
        x = tf.reduce_mean(x, axis=1)  # (batch_size, embed_dim)

        # 分类头输出
        x = self.classifier(x)

        return x


# 创建模型实例
model = VisionTransformer(img_size=224, patch_size=8, embed_dim=512, num_heads=8, num_layers=6, ff_dim=1024, num_classes=1000, activation=activations.swish)


