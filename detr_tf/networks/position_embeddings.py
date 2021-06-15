import numpy as np
import tensorflow as tf

class PositionEmbeddingSine(tf.keras.layers.Layer):
    # These are the default parameters used in the original project
    def __init__(self, num_pos_features=64, temperature=10000,
                 normalize=False, scale=None, eps=1e-6, center=False, **kwargs):
        super().__init__(**kwargs)

        self.num_pos_features = num_pos_features
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * np.pi
        self.scale = scale
        self.eps = eps
        self.center = center



    def call(self, mask):

        not_mask = tf.cast(mask == 0, tf.float32)

        y_embed_mask = tf.cumsum(not_mask, axis=1)
        x_embed_mask = tf.cumsum(not_mask, axis=2)
        y_embed_mask = tf.squeeze(y_embed_mask, axis=-1)
        x_embed_mask = tf.squeeze(x_embed_mask, axis=-1)

        #print("y_embed_mask", y_embed_mask.shape)
        #print("x_embed_mask", x_embed_mask.shape)

        x = tf.range(tf.shape(mask)[2]) + 1
        y = tf.range(tf.shape(mask)[1]) + 1
        x_embed, y_embed = tf.meshgrid(x, y)

        x_embed = tf.expand_dims(x_embed, axis=0)
        y_embed = tf.expand_dims(y_embed, axis=0)

        x_embed = tf.tile(x_embed, [tf.shape(mask)[0], 1, 1,])
        y_embed = tf.tile(y_embed, [tf.shape(mask)[0], 1, 1,])
        x_embed = tf.cast(x_embed, tf.float32)
        y_embed = tf.cast(y_embed, tf.float32)

        #print('x_embed', x_embed.shape)
        #print("y_embed", y_embed.shape)

        if self.normalize:
            if self.center:
                y_embed = y_embed-0.5
                x_embed = x_embed-0.5
            y_embed = y_embed / (y_embed_mask[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed_mask[:, :, -1:] + self.eps) * self.scale

        dim_t = tf.range(self.num_pos_features, dtype=tf.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_features)

        x_embed = tf.expand_dims(x_embed, axis=-1)
        y_embed = tf.expand_dims(y_embed, axis=-1)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t

        pos_x = tf.stack([tf.math.sin(pos_x[..., 0::2]),
                          tf.math.cos(pos_x[..., 1::2])], axis=4)

        pos_y = tf.stack([tf.math.sin(pos_y[..., 0::2]),
                          tf.math.cos(pos_y[..., 1::2])], axis=4)

        shape = [tf.shape(pos_x)[i] for i in range(3)] + [-1]
        pos_x = tf.reshape(pos_x, shape)
        pos_y = tf.reshape(pos_y, shape)

        pos_emb = tf.concat([pos_y, pos_x], axis=3)

        return pos_emb
