import tensorflow as tf

class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, patch_size, embed_dim, stride=16, activation=tf.keras.layers.ReLU()):
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