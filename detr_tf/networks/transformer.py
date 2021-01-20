import tensorflow as tf
from tensorflow.keras.layers import Dropout, Activation, LayerNormalization

from .custom_layers import Linear


class Transformer(tf.keras.Model):
    def __init__(self, model_dim=256, num_heads=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', normalize_before=False,
                 return_intermediate_dec=False, **kwargs):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        enc_norm = LayerNormalization(epsilon=1e-5, name='norm_pre') if normalize_before else None
        self.encoder = TransformerEncoder(model_dim, num_heads, dim_feedforward,
                                          dropout, activation, normalize_before, enc_norm,
                                          num_encoder_layers, name='encoder')

        dec_norm = LayerNormalization(epsilon=1e-5, name='norm')
        self.decoder = TransformerDecoder(model_dim, num_heads, dim_feedforward,
                                          dropout, activation, normalize_before, dec_norm,
                                          num_decoder_layers, name='decoder',
                                          return_intermediate=return_intermediate_dec)


    def call(self, source, mask, query_encoding, pos_encoding, training=False):

        batch_size, rows, cols = [tf.shape(source)[i] for i in range(3)]
        source = tf.reshape(source, [batch_size, -1, self.model_dim])
        source = tf.transpose(source, [1, 0, 2])



        pos_encoding = tf.reshape(pos_encoding, [batch_size, -1, self.model_dim])
        pos_encoding = tf.transpose(pos_encoding, [1, 0, 2])

        query_encoding = tf.expand_dims(query_encoding, axis=1)
        query_encoding = tf.tile(query_encoding, [1, batch_size, 1])

        mask = tf.reshape(mask, [batch_size, -1])

        target = tf.zeros_like(query_encoding)

        memory = self.encoder(source, source_key_padding_mask=mask,
                              pos_encoding=pos_encoding, training=training)
        hs = self.decoder(target, memory, memory_key_padding_mask=mask,
                          pos_encoding=pos_encoding, query_encoding=query_encoding,
                          training=training)

        hs = tf.transpose(hs, [0, 2, 1, 3])
        memory = tf.transpose(memory, [1, 0, 2])
        memory = tf.reshape(memory, [batch_size, rows, cols, self.model_dim])

        return hs, memory


class TransformerEncoder(tf.keras.Model):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False, norm=None,
                 num_encoder_layers=6, **kwargs):
        super().__init__(**kwargs)

        self.enc_layers = [EncoderLayer(model_dim, num_heads, dim_feedforward,
                                        dropout, activation, normalize_before,
                                        name='layer_%d'%i)
                           for i in range(num_encoder_layers)]
        
        self.norm = norm


    def call(self, source, mask=None, source_key_padding_mask=None,
             pos_encoding=None, training=False):
        x = source


        for layer in self.enc_layers:
            x = layer(x, source_mask=mask, source_key_padding_mask=source_key_padding_mask,
                      pos_encoding=pos_encoding, training=training)

        if self.norm:
            x = self.norm(x)

        return x


class TransformerDecoder(tf.keras.Model):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False, norm=None,
                 num_decoder_layers=6, return_intermediate=False, **kwargs):
        super().__init__(**kwargs)

        self.dec_layers = [DecoderLayer(model_dim, num_heads, dim_feedforward,
                                        dropout, activation, normalize_before,
                                        name='layer_%d'%i)
                           for i in range(num_decoder_layers)]

        self.norm = norm
        self.return_intermediate = return_intermediate


    def call(self, target, memory, target_mask=None, memory_mask=None,
             target_key_padding_mask=None, memory_key_padding_mask=None,
             pos_encoding=None, query_encoding=None, training=False):

        x = target
        intermediate = []


        for layer in self.dec_layers:
            x = layer(x, memory,
                      target_mask=target_mask,
                      memory_mask=memory_mask,
                      target_key_padding_mask=target_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos_encoding=pos_encoding,
                      query_encoding=query_encoding)

            if self.return_intermediate:
                if self.norm:
                    intermediate.append(self.norm(x))
                else:
                    intermediate.append(x)

        if self.return_intermediate:
            return tf.stack(intermediate, axis=0)

        if self.norm:
            x = self.norm(x)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                            name='self_attn')

        self.dropout = Dropout(dropout)
        self.activation = Activation(activation)

        self.linear1 = Linear(dim_feedforward, name='linear1')
        self.linear2 = Linear(model_dim, name='linear2')

        self.norm1 = LayerNormalization(epsilon=1e-5, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-5, name='norm2')

        self.normalize_before = normalize_before


    def call(self, source, source_mask=None, source_key_padding_mask=None,
             pos_encoding=None, training=False):


        if pos_encoding is None:
            query = key = source
        else:
            query = key = source + pos_encoding

        attn_source = self.self_attn((query, key, source), attn_mask=source_mask,
                                     key_padding_mask=source_key_padding_mask,
                                     need_weights=False)
        source += self.dropout(attn_source, training=training)
        source = self.norm1(source)

        x = self.linear1(source)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        source += self.dropout(x, training=training)
        source = self.norm2(source)
        
        return source



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu', normalize_before=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                            name='self_attn')
        self.multihead_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                                 name='multihead_attn')

        self.dropout = Dropout(dropout)
        self.activation = Activation(activation)

        self.linear1 = Linear(dim_feedforward, name='linear1')
        self.linear2 = Linear(model_dim, name='linear2')

        self.norm1 = LayerNormalization(epsilon=1e-5, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-5, name='norm2')
        self.norm3 = LayerNormalization(epsilon=1e-5, name='norm3')

        self.normalize_before = normalize_before


    def call(self, target, memory, target_mask=None, memory_mask=None,
             target_key_padding_mask=None, memory_key_padding_mask=None,
             pos_encoding=None, query_encoding=None, training=False):

        query_tgt = key_tgt = target + query_encoding
        attn_target = self.self_attn((query_tgt, key_tgt, target), attn_mask=target_mask,
                                    key_padding_mask=target_key_padding_mask,
                                    need_weights=False)
        target += self.dropout(attn_target, training=training)
        target = self.norm1(target)

        query_tgt = target + query_encoding
        key_mem = memory + pos_encoding
        
        attn_target2 = self.multihead_attn((query_tgt, key_mem, memory), attn_mask=memory_mask,
                                           key_padding_mask=memory_key_padding_mask,
                                           need_weights=False)
        target += self.dropout(attn_target2, training=training)
        target = self.norm2(target)

        x = self.linear1(target)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        x = self.linear2(x)
        target += self.dropout(x, training=training)
        target = self.norm3(target)
        
        return target


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.dropout = Dropout(rate=dropout)
        

    def build(self, input_shapes):
        in_dim = sum([shape[-1] for shape in input_shapes[:3]])

        self.in_proj_weight = self.add_weight(
            name='in_proj_kernel', shape=(in_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.in_proj_bias = self.add_weight(
            name='in_proj_bias', shape=(in_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_weight = self.add_weight(
            name='out_proj_kernel', shape=(self.model_dim, self.model_dim),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )
        self.out_proj_bias = self.add_weight(
            name='out_proj_bias', shape=(self.model_dim,),
            initializer=tf.keras.initializers.GlorotUniform(), dtype=tf.float32, trainable=True
        )




        #self.in_proj_weight = tf.Variable(
        #    tf.zeros((in_dim, self.model_dim), dtype=tf.float32), name='in_proj_kernel')
        #self.in_proj_bias = tf.Variable(tf.zeros((in_dim,), dtype=tf.float32),
        #                                name='in_proj_bias')

        #self.out_proj_weight = tf.Variable(
        #    tf.zeros((self.model_dim, self.model_dim), dtype=tf.float32), name='out_proj_kernel')
        #self.out_proj_bias = tf.Variable(
        #    tf.zeros((self.model_dim,), dtype=tf.float32), name='out_proj_bias')



    def call(self, inputs, attn_mask=None, key_padding_mask=None,
             need_weights=True, training=False):

        query, key, value = inputs

        batch_size = tf.shape(query)[1]
        target_len = tf.shape(query)[0]
        source_len = tf.shape(key)[0]

        W = self.in_proj_weight[:self.model_dim, :]
        b = self.in_proj_bias[:self.model_dim]

        WQ = tf.matmul(query, W, transpose_b=True) + b

        W = self.in_proj_weight[self.model_dim:2*self.model_dim, :]
        b = self.in_proj_bias[self.model_dim:2*self.model_dim]
        WK = tf.matmul(key, W, transpose_b=True) + b

        W = self.in_proj_weight[2*self.model_dim:, :]
        b = self.in_proj_bias[2*self.model_dim:]
        WV = tf.matmul(value, W, transpose_b=True) + b

        WQ *= float(self.head_dim) ** -0.5
        WQ = tf.reshape(WQ, [target_len, batch_size * self.num_heads, self.head_dim])
        WQ = tf.transpose(WQ, [1, 0, 2])
        
        WK = tf.reshape(WK, [source_len, batch_size * self.num_heads, self.head_dim])
        WK = tf.transpose(WK, [1, 0, 2])

        WV = tf.reshape(WV, [source_len, batch_size * self.num_heads, self.head_dim])
        WV = tf.transpose(WV, [1, 0, 2])
        
        attn_output_weights = tf.matmul(WQ, WK, transpose_b=True)

        if attn_mask is not None:
            attn_output_weights += attn_mask

        """
        if key_padding_mask is not None:
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size, self.num_heads, target_len, source_len])

            key_padding_mask = tf.expand_dims(key_padding_mask, 1)
            key_padding_mask = tf.expand_dims(key_padding_mask, 2)
            key_padding_mask = tf.tile(key_padding_mask, [1, self.num_heads, target_len, 1])

            #print("before attn_output_weights", attn_output_weights.shape)
            attn_output_weights = tf.where(key_padding_mask,
                                           tf.zeros_like(attn_output_weights) + float('-inf'),
                                           attn_output_weights)
            attn_output_weights = tf.reshape(attn_output_weights,
                                [batch_size * self.num_heads, target_len, source_len])
        """


        attn_output_weights = tf.nn.softmax(attn_output_weights, axis=-1)
        attn_output_weights = self.dropout(attn_output_weights, training=training)

        attn_output = tf.matmul(attn_output_weights, WV)
        attn_output = tf.transpose(attn_output, [1, 0, 2])
        attn_output = tf.reshape(attn_output, [target_len, batch_size, self.model_dim])
        attn_output = tf.matmul(attn_output, self.out_proj_weight,
                                transpose_b=True) + self.out_proj_bias

        if need_weights:
            attn_output_weights = tf.reshape(attn_output_weights,
                            [batch_size, self.num_heads, target_len, source_len])
            # Retrun the average weight over the heads
            avg_weights = tf.reduce_mean(attn_output_weights, axis=1)
            return attn_output, avg_weights
        
        return attn_output
