import tensorflow as tf
from tensorflow.keras.layers import Dropout, Activation, LayerNormalization
import math
from .custom_layers import Linear
from .transformer import MultiHeadAttention


USE_CUDA_MS_DEFORM_IM2COL = True

if USE_CUDA_MS_DEFORM_IM2COL:
    from detr_tf.custom_ops.ms_deform_attn import ms_deform_im2col
else:
    from detr_tf.custom_ops.ms_deform_attn.ms_deform_attn import MSDeformAttnFunction

class DeformableTransformer(tf.keras.layers.Layer):
    def __init__(self,
                 layer_position_embedding_sine,
                 level_embed,
                 class_embed,
                 bbox_embed,
                 query_embed_layer=None,
                 model_dim=256,
                 num_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 num_sampling_points=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 activation='relu',
                 return_intermediate_dec=False,
                 init_query_embedding=False,
                 use_track_query=False,
                 refine_bbox=False,
                 multiscale=True,
                 train_encoder=True,
                 **kwargs):

        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.layer_position_embedding_sine = layer_position_embedding_sine
        self.query_embed_layer = query_embed_layer

        self.level_embed = level_embed

        self.class_embed = class_embed
        self.bbox_embed = bbox_embed

        self.init_query_embedding = init_query_embedding

        self.multiscale = multiscale

        self.encoder = DeformableEncoder(model_dim, num_heads, dim_feedforward,
                                        dropout, activation,
                                        num_encoder_layers, num_sampling_points=num_sampling_points, name='encoder', trainable=train_encoder)

        self.decoder = DeformableDecoder(class_embed, bbox_embed, model_dim, num_heads, dim_feedforward,
                                          dropout, activation,
                                          num_decoder_layers,
                                          name='decoder',
                                          num_sampling_points=num_sampling_points, refine_bbox=refine_bbox,
                                          return_intermediate=return_intermediate_dec, use_track_query=use_track_query)


        if self.init_query_embedding:
            raise NotImplementedError()
            self.query_encoding = self.add_weight(name='query_embedding', shape=(100, 256),
                                     initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        else:
            self.init_query_embedding = init_query_embedding

        self.reference_points = Linear(2,
                kernel_initializer=tf.keras.initializers.Zeros(),
                bias_initializer=tf.keras.initializers.Zeros(),
                name="reference_points")

    def get_reference_points(self, spatial_shapes):
        reference_points_list = []
        for lvl, (H_W_) in enumerate(spatial_shapes):
            H_, W_ = tf.unstack(H_W_)
            ref_y, ref_x = tf.meshgrid(tf.linspace(0.5, tf.cast(H_, tf.float32) - 0.5, H_),
                                       tf.linspace(0.5, tf.cast(W_, tf.float32) - 0.5, W_), indexing='ij')

            ref_y = ref_y / tf.cast(H_, tf.float32)
            ref_x = ref_x / tf.cast(W_, tf.float32)

            ref = tf.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        # L, (H, W, 2)
        return reference_points_list

    def call(self, source, mask, track_query=None, track_query_mask=None, track_query_reference_points=None, training=False):

        N = tf.shape(source[0])[0]


        pos_encoding = list(self.layer_position_embedding_sine(m) for m in mask)

        if self.init_query_embedding:
            # Panoptic embedding
            query_encoding = self.query_encoding
        else:
            # Detection embedding
            query_encoding = self.query_embed_layer(None)

        query_encoding, target = tf.split(query_encoding, 2, axis=1)
        query_encoding = tf.expand_dims(query_encoding, axis=1)

        if self.level_embed is not None:
            level_embed = self.level_embed(None)
            lvl_pos_embed = list(level_embed[lvl, None, :] + tf.reshape(p, (N, -1, self.model_dim) ) for lvl, p in enumerate(pos_encoding) ) # N, (H*W), C   +   1, 1, C
            lvl_pos_embed_flatten = tf.concat(lvl_pos_embed, axis=1) # N, sum_L(H*W), C
        else:
            lvl_pos_embed_flatten = None

        # L, 2
        input_spatial_shapes = list(tf.shape(src)[1:3] for src in source)
        encoder_reference_points = self.get_reference_points(input_spatial_shapes)
        input_spatial_shapes = tf.stack(input_spatial_shapes, 0)

        # L,
        input_level_start_index = tf.math.reduce_prod(input_spatial_shapes, axis=1)
        input_level_start_index = tf.math.cumsum(input_level_start_index, axis=0, exclusive=True)

        # L, (H, W, 2) -> L, (1, H*W, 2)
        encoder_reference_points = list( tf.reshape(rp_l, (1, -1, 2)) for rp_l in encoder_reference_points)
        encoder_reference_points = tf.concat(encoder_reference_points, axis=1)



        #Flatten sources
        source = [tf.reshape(s, (N, -1, self.model_dim) ) for s in source]
        source = tf.concat(source, axis=1)
        memory = self.encoder(source, encoder_reference_points, source_key_padding_mask=mask,
                            pos_encoding=lvl_pos_embed_flatten,
                            training=training,
                            source_spatial_shapes=input_spatial_shapes,
                            source_level_start_index=input_level_start_index)


        decoder_reference_points = tf.math.sigmoid(self.reference_points(query_encoding))
        decoder_reference_points = tf.tile(decoder_reference_points, [1, N, 1])

        if track_query_reference_points is not None:
            decoder_reference_points = tf.concat([track_query_reference_points, decoder_reference_points], axis=0)

        target = tf.reshape(target, (300, 1, self.model_dim) )
        target = tf.tile(target, [1, N, 1])


        hs, reference_points = self.decoder(target, memory, decoder_reference_points, memory_key_padding_mask=mask,
                          pos_encoding=lvl_pos_embed_flatten, query_encoding=query_encoding,
                          track_query=track_query, track_query_mask=track_query_mask,
                          memory_spatial_shapes=input_spatial_shapes,
                          memory_level_start_index=input_level_start_index,
                          training=training)

        return tf.transpose(hs, [0, 2, 1, 3]), tf.transpose(memory, (1, 0, 2)), tf.transpose(reference_points, [0, 2, 1, 3])


class DeformableEncoder(tf.keras.layers.Layer):
    def __init__(self, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu',
                 num_encoder_layers=6, num_sampling_points=4, **kwargs):
        super().__init__(**kwargs)

        self.enc_layers = [DeformableEncoderLayer(model_dim, num_heads, num_sampling_points, dim_feedforward,
                                        dropout, activation,
                                        name='layer_%d'%i)
                           for i in range(num_encoder_layers)]


    def call(self, source, reference_points, mask=None, source_key_padding_mask=None,
             pos_encoding=None, track_query=None,
             source_spatial_shapes=None, source_level_start_index=None, training=False):
        x = source

        for l_id, layer in enumerate(self.enc_layers):
            x = layer(x, reference_points, source_mask=mask, source_key_padding_mask=source_key_padding_mask,
                      pos_encoding=pos_encoding, input_spatial_shapes=source_spatial_shapes, input_level_start_index=source_level_start_index, training=training)

        return x


class DeformableDecoder(tf.keras.layers.Layer):
    def __init__(self, class_embed, bbox_embed, model_dim=256, num_heads=8, dim_feedforward=2048,
                 dropout=0.1, activation='relu',
                 num_decoder_layers=6, num_sampling_points=4, return_intermediate=False, use_track_query=False, refine_bbox=False, **kwargs):
        super().__init__(**kwargs)

        self.dec_layers = [DeformableDecoderLayer(model_dim, num_heads, num_sampling_points, dim_feedforward,
                                        dropout, activation, use_track_query=use_track_query,
                                        name='layer_%d'%i)
                           for i in range(num_decoder_layers)]

        self.class_embed = class_embed
        self.bbox_embed = bbox_embed

        self.refine_bbox = refine_bbox
        self.return_intermediate = return_intermediate


    def call(self, target, memory, reference_points, target_mask=None, memory_mask=None,
             target_key_padding_mask=None, memory_key_padding_mask=None, memory_spatial_shapes=None, memory_level_start_index=None,
             pos_encoding=None, query_encoding=None, track_query=None, track_query_mask=None, training=False):


        x = target
        intermediate = []
        intermediate_reference_points = []

        new_reference_points = reference_points

        for l_id, layer in enumerate(self.dec_layers):
            # if the tracking is not use
            # track_query we'll simply stay None.
            x, track_query = layer(x, memory, reference_points,
                      target_mask=target_mask,
                      memory_mask=memory_mask,
                      target_key_padding_mask=target_key_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask,
                      pos_encoding=pos_encoding,
                      track_query=track_query,
                      track_query_mask=track_query_mask,
                      query_encoding=query_encoding,
                      memory_spatial_shapes=memory_spatial_shapes,
                      memory_level_start_index=memory_level_start_index)

            if track_query is not None:
                out = tf.concat([track_query, x], axis=0)
            else:
                out = x

            tmp = self.bbox_embed[l_id](out)
            if self.refine_bbox:
                new_reference_points = inverse_sigmoid(new_reference_points)
            else:
                new_reference_points = inverse_sigmoid(reference_points)

            if new_reference_points.shape[-1] == 4:
                new_reference_points = tmp + new_reference_points
            elif new_reference_points.shape[-1] == 2:
                xy = tmp[..., :2] + new_reference_points
                hw = tmp[..., 2:]
                new_reference_points = tf.concat([xy, hw], axis=-1)
            else:
                raise ValueError()


            new_reference_points = tf.math.sigmoid(new_reference_points)

            if self.refine_bbox:
                reference_points = tf.stop_gradient(new_reference_points)

            if self.return_intermediate:
                intermediate.append(out)
                intermediate_reference_points.append(new_reference_points)


        if self.return_intermediate:
            return tf.stack(intermediate, axis=0), tf.stack(intermediate_reference_points, axis=0)

        return out, reference_points


class DeformableEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim=256, num_heads=8, num_sampling_points=4, dim_feedforward=2048,
                 dropout=0.1, activation='relu',
                 **kwargs):
        super().__init__(**kwargs)

        self.self_attn =  MSDeformableAttention(model_dim, num_heads, num_sampling_points, dropout=dropout,
                                            name='self_attn')

        self.dropout = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = Activation(activation)

        self.model_dim = model_dim

        self.linear1 = Linear(dim_feedforward, name='linear1')
        self.linear2 = Linear(model_dim, name='linear2')

        self.norm1 = LayerNormalization(epsilon=1e-5, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-5, name='norm2')


    def call(self, source, reference_points, source_mask=None, source_key_padding_mask=None,
             pos_encoding=None, input_spatial_shapes=None, input_level_start_index=None, training=False):
        """
        :param source                       (N, H, W, C)
        :param pos_encoding                 (1, sum_L(H*W), C)
        :param reference_points             (H, W, 2)

        :return output                      (N, sum_L(H*W), C)
        """

        N = tf.shape(source[0])[0]
        C = self.model_dim

        if pos_encoding is None:
            query = source
        else:
            query = source + pos_encoding

        # # Multi-scale level embedding L, (N, H, W, C)
        # query = list(q + level_embed[lvl, None, None, :] for lvl, q in enumerate(query))

        # # Flatten ¤¤¤¤
        # # L, (N, H*W, C)
        # query = list(tf.reshape(q, (N, -1, C) ) for q in query)
        # # (N, sum_L{H_*W_}, C)
        # query = tf.concat(query, axis=1)

        # (N, Length_{query}, C)

        #print("query", query.shape)
        attn_source = self.self_attn(query, reference_points, source, input_spatial_shapes, input_level_start_index)
        #print("attn_source", attn_source.shape)
        # src = list(tf.reshape(s, (N, -1, C) ) for s in source)
        # # (N, sum_L{H_*W_}, C)
        # src = tf.concat(src, axis=1)

        source += self.dropout(attn_source, training=training)
        source = self.norm1(source)

        #forward_ffn
        x = self.linear1(source)
        x = self.activation(x)
        x = self.dropout2(x, training=training)
        x = self.linear2(x)
        source += self.dropout3(x, training=training)
        source = self.norm2(source)

        # #Unflatten ¤¤¤¤
        # split_size = list(iss[0]*iss[1] for iss in input_spatial_shapes)
        # # L, (N, H*W, 2)
        # src = tf.split(src, split_size, axis=1)
        # # L, (N, H, W, 2)
        # src = list(tf.reshape(el, (N, iss[0], iss[1], C) ) for iss, el in zip(input_spatial_shapes, src))

        return source



class DeformableDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, model_dim=256, num_heads=8, num_sampling_points=4, dim_feedforward=2048,
                 dropout=0.1, activation='relu', use_track_query=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout,
                                            name='self_attn')
        self.cross_attn = MSDeformableAttention(model_dim, num_heads, dropout=dropout,
                                                 name='cross_attn')

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self.dropout4 = Dropout(dropout)

        self.activation = Activation(activation)

        self.linear1 = Linear(dim_feedforward, name='linear1')
        self.linear2 = Linear(model_dim, name='linear2')

        self.norm1 = LayerNormalization(epsilon=1e-5, name='norm1')
        self.norm2 = LayerNormalization(epsilon=1e-5, name='norm2')
        self.norm3 = LayerNormalization(epsilon=1e-5, name='norm3')

        self.use_track_query = use_track_query

        # if self.use_track_query:
        #     self.dropout = Dropout(dropout)
        #     self.track_query_norm = LayerNormalization(epsilon=1e-5, name='track_query_norm')
        #     self.track_query_self_attn = MultiHeadAttention(model_dim, num_heads, dropout=dropout, name='track_query_self_attn')


    def call(self, target, memory, reference_points, target_mask=None, memory_mask=None,
             target_key_padding_mask=None, memory_key_padding_mask=None, memory_spatial_shapes=None, memory_level_start_index=None,
             pos_encoding=None, track_query=None, track_query_mask=None, query_encoding=None, level_embed=None, training=False):


        if track_query is not None:
            # track_query_query = track_query
            # track_query_key = track_query
            # track_query_target = track_query

            if target_key_padding_mask is None:
                target_shape = tf.shape(target)
                target_key_padding_mask = tf.zeros((target_shape[1], target_shape[0]))

            # track_query_attn_target = self.track_query_self_attn((track_query_query, track_query_key, track_query_target), key_padding_mask=track_query_mask, need_weights=False)

            # track_query_target += self.dropout(track_query_attn_target, training=training)
            # track_query_target = self.track_query_norm(track_query_target)
            nb_trackquery = tf.shape(track_query)[0]

            # Pad on the left the original query
            query_encoding = tf.pad(query_encoding, [[nb_trackquery, 0], [0, 0], [0, 0]], "CONSTANT" )
            # Concat with the track query on the left
            target = tf.concat([track_query, target], axis=0)
            target_key_padding_mask = tf.concat([track_query_mask, target_key_padding_mask], axis=1)

        # If we use the track query, the query encoding is now padded with zeros for the track queries
        # query_tgt = target + query_encoding
        query_tgt = key_tgt = target + query_encoding

        attn_target = self.self_attn((query_tgt, key_tgt, target), attn_mask=target_mask,
                                    key_padding_mask=target_key_padding_mask,
                                    need_weights=False)

        target += self.dropout2(attn_target, training=training)
        target = self.norm2(target)

        query_tgt = target + query_encoding

        query_tgt = tf.transpose(query_tgt, (1, 0, 2) )
        reference_points = tf.transpose(reference_points, (1, 0, 2) )

        attn_target2 = self.cross_attn(query_tgt, reference_points, memory,
                                       input_spatial_shapes=memory_spatial_shapes, input_level_start_index=memory_level_start_index)
        attn_target2 = tf.transpose(attn_target2, (1, 0, 2) )

        target += self.dropout1(attn_target2, training=training)
        target = self.norm1(target)

        x = self.linear1(target)
        x = self.activation(x)
        x = self.dropout3(x, training=training)
        x = self.linear2(x)
        target += self.dropout4(x, training=training)
        target = self.norm3(target)

        if track_query is not None:
            n_track_query = target[:nb_trackquery]
            target = target[nb_trackquery:]
            return target, n_track_query
        else:
            return target, None


class SamplingOffsetBiasInitializer(tf.keras.initializers.Initializer):

  def __init__(self, num_heads, num_level, n_points):
    self.num_heads = num_heads
    self.num_level = num_level
    self.n_points = n_points

  def __call__(self, shape, dtype=None, **kwargs):
    thetas = tf.range(self.num_heads, dtype=tf.float32) * (2.0 * math.pi / self.num_heads)
    grid_init = tf.stack([tf.math.cos(thetas), tf.math.sin(thetas)], axis=-1)
    grid_init = grid_init / tf.math.reduce_max(tf.abs(grid_init), axis=-1, keepdims=True)[0]
    grid_init = tf.reshape(grid_init, (self.num_heads, 1, 1, 2) )
    # self.num_heads, self.num_level, self.n_points, 2
    grid_init = tf.tile(grid_init, (1, self.num_level, self.n_points, 1) )

    scaling = tf.range(self.n_points, dtype = tf.float32) + 1.0
    scaling = tf.reshape(scaling, (1, 1, self.n_points , 1) )
    grid_init = grid_init * scaling

    grid_init = tf.reshape(grid_init, (-1,))

    return grid_init



class MSDeformableAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_heads, num_sampling_points = 4, num_level=4, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.model_dim = model_dim
        self.num_heads = num_heads

        self.num_level = num_level
        self.num_sampling_points = num_sampling_points

        assert model_dim % num_heads == 0
        self.head_dim = model_dim // num_heads

        self.dropout = Dropout(rate=dropout)

        self.im2col_step = 64


    def build(self, input_shapes):

        self.sampling_offsets = Linear(self.num_heads * self.num_level * self.num_sampling_points * 2,
                                                        kernel_initializer=tf.keras.initializers.Zeros(),
                                                        bias_initializer=SamplingOffsetBiasInitializer(self.num_heads, self.num_level, self.num_sampling_points),
                                                        name="sampling_offsets")

        self.attention_weights = Linear(self.num_heads * self.num_level * self.num_sampling_points,
                                                        kernel_initializer=tf.keras.initializers.Zeros(),
                                                        bias_initializer=tf.keras.initializers.Zeros(),
                                                        name="attention_weights")

        self.value_proj = Linear(self.model_dim, bias_initializer=tf.keras.initializers.Zeros(), name="value_proj")

        self.output_proj = Linear(self.model_dim, name="output_proj")


    def call(self, query, reference_points, inputs, input_spatial_shapes=None, input_level_start_index=None, input_padding_mask=None, training=False):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, 4), add additional (w, h) to form reference boxes

        :param inputs                      (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
            or
        :param inputs                      lvl, (N, H_l, W_l, C)

        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """

        #debug purpose
        unstack_size = list(iss[0]*iss[1] for iss in tf.unstack(input_spatial_shapes, axis=0))
        unstack_shape = list( (iss[0], iss[1]) for iss in tf.unstack(input_spatial_shapes, axis=0))

        N, Len_q, C = tf.unstack(tf.shape(query))

        N, Len_in, _ = tf.unstack(tf.shape(inputs))
        value = self.value_proj(inputs)
        value = tf.reshape(value, (N, Len_in, self.num_heads, self.head_dim))


        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = tf.reshape(sampling_offsets, (N, Len_q, self.num_heads, self.num_level, self.num_sampling_points, 2) )


        attention_weights = self.attention_weights(query)
        attention_weights = tf.reshape(attention_weights, (N, Len_q, self.num_heads,  self.num_level * self.num_sampling_points) )
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = tf.reshape(attention_weights, (N, Len_q, self.num_heads,  self.num_level, self.num_sampling_points) )


        # (N, Len_q, num_heads, num_level, num_sampling_points, _)
        if reference_points.shape[-1] == 2:
               offset_normalizer = tf.cast(tf.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1), tf.float32)
               sampling_locations = reference_points[:, :, None, None, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
                sampling_locations = reference_points[:, :, None, None, None, :2] + sampling_offsets / self.num_sampling_points * reference_points[:, :, None, None, None, 2:] * 0.5
        else:
            raise ValueError(f"reference_points shape must be defined, got {reference_points.shape[-1]}")

        if USE_CUDA_MS_DEFORM_IM2COL:
            # Flatten and call custom op !
            output = ms_deform_im2col(
                value, # (N, Len_in, n_heads, d_model#n_heads)
                input_spatial_shapes,  # (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
                input_level_start_index, # (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
                sampling_locations, # (N, Len_q, n_heads, n_levels, n_points, 2)
                attention_weights # (N, Len_q, num_heads,  n_level, num_sampling_points)
            )
        else:
            #Unflatten
            value = tf.split(value, num_or_size_splits=unstack_size, axis=1)
            value = list(tf.reshape(v, (N, shape[0], shape[1], self.num_heads, self.head_dim) ) for v, shape in zip(value, unstack_shape) )

            sampling_loc = tf.unstack(sampling_locations, axis=3) #(N, Len_q, n_heads, num_sampling_points)

            output = MSDeformAttnFunction(value, sampling_loc, attention_weights)


        output = self.output_proj(output)


        return output

def inverse_sigmoid(x, eps=1e-5):
    x = tf.clip_by_value(x, 0.0, 1.0)
    x1 = tf.clip_by_value(x, eps, 1.0)

    x2 = (1 - x)
    x2 = tf.clip_by_value(x2, eps, 1.0)
    return tf.math.log(x1/x2)
