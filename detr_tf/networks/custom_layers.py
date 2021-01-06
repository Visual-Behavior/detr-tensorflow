import tensorflow as tf


class FrozenBatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps


    def build(self, input_shape):
        self.weight = self.add_weight(name='weight', shape=[input_shape[-1]],
                                      initializer=tf.keras.initializers.GlorotUniform(), trainable=False)
        self.bias = self.add_weight(name='bias', shape=[input_shape[-1]],
                                    initializer=tf.keras.initializers.GlorotUniform(), trainable=False)
        self.running_mean = self.add_weight(name='running_mean', shape=[input_shape[-1]],
                                            initializer='zeros', trainable=False)
        self.running_var = self.add_weight(name='running_var', shape=[input_shape[-1]],
                                           initializer='ones', trainable=False)


    def call(self, x):
        scale = self.weight * tf.math.rsqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift


    def compute_output_shape(self, input_shape):
        return input_shape


class Linear(tf.keras.layers.Layer):
    '''
    Use this custom layer instead of tf.keras.layers.Dense to allow
    loading converted PyTorch Dense weights that have shape (output_dim, input_dim)
    '''
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim


    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=[self.output_dim, input_shape[-1]],
                                      initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=[self.output_dim],
                                    initializer=tf.keras.initializers.GlorotUniform(), trainable=True)

    def call(self, x):
        return tf.matmul(x, self.kernel, transpose_b=True) + self.bias


    def compute_output_shape(self, input_shape):
        return input_shape.as_list()[:-1] + [self.output_dim]


class FixedEmbedding(tf.keras.layers.Layer):
    def __init__(self, embed_shape, **kwargs):
        super().__init__(**kwargs)
        self.embed_shape = embed_shape

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel', shape=self.embed_shape,
                                 initializer=tf.keras.initializers.GlorotUniform(), trainable=True)

    def call(self, x=None):
        return self.w
