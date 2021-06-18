import pickle
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import os
import math
import json
from pathlib import Path
import tensorflow_addons as tfa
import functools
import collections

from detr_tf.networks.deformable_transformer import DeformableTransformer
from detr_tf.networks.transformer import MultiHeadAttention
from detr_tf.networks.resnet_backbone import ResNet50Backbone
from detr_tf.networks.custom_layers import Linear, FixedEmbedding, ScaleLevelEmbedding, MLP
from detr_tf.networks.position_embeddings import PositionEmbeddingSine
from detr_tf.networks.transformer import Transformer
from detr_tf.networks.weights import load_weights

class DeformableDETR(tf.keras.Model):
    def __init__(self,
                 model_dim=256,
                 num_classes=91,
                 num_queries=300,
                 num_sampling_points=4,
                 backbone=None,
                 pos_encoder=None,
                 transformer=None,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 return_intermediate_dec=True,
                 init_query_embedding=False,
                 batch_size=None,
                 use_mask_bn=False,
                 refine_bbox=True,
                 multiscale=True,
                 train_encoder=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.backbone = ResNet50Backbone(name='backbone')

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=model_dim // 2, normalize=True, center=True)

        self.query_embed = FixedEmbedding((num_queries, model_dim*2), name='query_embed')
        self.level_embed = ScaleLevelEmbedding(4, model_dim, name="level_embed", trainable=train_encoder)


        self.multiscale = multiscale


        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed = list(
            Linear(num_classes,
                bias_initializer=tf.keras.initializers.Constant(bias_value),
                name=f'class_embed_{i}')
            for i in range(num_decoder_layers))

        self.bbox_embed = list(
            MLP(model_dim,
                4,
                kernel_initializer=tf.keras.initializers.Zeros(),
                bias_initializer=tf.keras.initializers.Zeros(),
                name=f'bbox_embed_{i}')
            for i in range(num_decoder_layers))

        # hack to force shared weight (different from pytorch cloning approach)
        if not refine_bbox:
            self.class_embed = [self.class_embed[0] for _ in range(num_decoder_layers)]
            self.bbox_embed = [self.bbox_embed[0] for _ in range(num_decoder_layers)]

        self.transformer = transformer or DeformableTransformer(
                query_embed_layer=self.query_embed,
                level_embed=self.level_embed,
                layer_position_embedding_sine=self.pos_encoder,
                model_dim=model_dim,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                num_sampling_points=num_sampling_points,
                return_intermediate_dec=return_intermediate_dec,
                init_query_embedding=init_query_embedding,
                class_embed=self.class_embed,
                bbox_embed=self.bbox_embed,
                refine_bbox=refine_bbox,
                train_encoder=train_encoder,
                name='transformer'
        )
        self.model_dim = model_dim

        layer_norm = functools.partial(tfa.layers.GroupNormalization, groups=32, epsilon=1e-05) #tf.keras.layers.BatchNormalization


        self.input_proj_0 = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj/0/0', trainable=train_encoder)
        self.input_proj_gn_0 = layer_norm(name="input_proj_gn/0/1", trainable=train_encoder)

        if multiscale:
            self.input_proj_1 = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj/1/0', trainable=train_encoder)
            self.input_proj_2 = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj/2/0', trainable=train_encoder)
            self.input_proj_3 = tf.keras.layers.Conv2D(self.model_dim, kernel_size=3, strides=2, name='input_proj/3/0', trainable=train_encoder)

            self.input_proj_gn_1 = layer_norm(name="input_proj_gn/1/1", trainable=train_encoder)
            self.input_proj_gn_2 = layer_norm(name="input_proj_gn/2/1", trainable=train_encoder)
            self.input_proj_gn_3 = layer_norm(name="input_proj_gn/3/1", trainable=train_encoder)

        #self.activation = tf.keras.layers.ReLU()

        self.num_decoder_layers = num_decoder_layers


    def call(self, inp, training=False, post_process=False):
        x = inp
        backbone_outputs = self.backbone(x, training=training)
        x2, x1, x0, _ = backbone_outputs

        if self.multiscale:
            src_proj_outputs = [self.input_proj_gn_0(self.input_proj_0(x0)), \
                            self.input_proj_gn_1(self.input_proj_1(x1)), \
                            self.input_proj_gn_2(self.input_proj_2(x2)), \
                            self.input_proj_gn_3(tf.keras.layers.ZeroPadding2D(1)(self.input_proj_3(x2)))]
        else:
            src_proj_outputs = [self.input_proj_gn_0(self.input_proj_0(x2))]

        masks = list(tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1], tf.float32) for x in src_proj_outputs)

        decoder, encoder, outputs_coord = self.transformer(src_proj_outputs,
                                            masks,
                                            training=training)

        outputs_class = list(class_embed(x) for class_embed, x in zip(self.class_embed, tf.split(decoder, self.num_decoder_layers)) )

        output = {'bbox_pred_logits': outputs_class[-1],
                  'bbox_pred_boxes': outputs_coord[-1]}

        if post_process:
            output = self.post_process(output)

        a = self.query_embed(None)

        return output


class DetrClassHead(tf.keras.layers.Layer):

    def __init__(self, detr:DeformableDETR, include_top, nb_class=None, refine_bbox=False, **kwargs):
        """
        """
        super().__init__(name="detr_class_head", **kwargs)
        self.include_top = include_top
        if self.include_top:
            if refine_bbox:
                self.layer_class_embed = list(detr.get_layer(f'class_embed_{i}') for i in range(detr.num_decoder_layers))
            else:
                #shared weights
                self.layer_class_embed = list(detr.get_layer(f'class_embed_0') for _ in range(detr.num_decoder_layers) )
        else:
            # Setup the new layers
            if refine_bbox:
                self.layer_class_embed = list(tf.keras.layers.Dense(nb_class, name=f"class_embed_{i}") for i in range(detr.num_decoder_layers) )
            else:
                layer = tf.keras.layers.Dense(nb_class, name=f"class_embed_0")
                self.layer_class_embed = list(layer for i in range(detr.num_decoder_layers) )

    def call(self, decoder_state):
        outputs = {}

        # Output class
        outputs_class = [l(s) for l, s in zip(self.layer_class_embed, tf.unstack(decoder_state, axis=0))]

        outputs = {'bbox_pred_logits': outputs_class[-1]}

        outputs["bbox_aux"] = []
        for out_class in outputs_class:
            outputs["bbox_aux"].append({
                "bbox_pred_logits": out_class
            })

        return outputs


    def build(self, input_shape=None, **kwargs):
        super().build(input_shape, **kwargs)



def get_detr_core(detr, backbone, model_dim, tf_backbone=False, multiscale=True):
    """ DETR Core is made of the backbone and the transformer part without the
    heads
    """

    layer_transformer = detr.get_layer("transformer")

    #### Set ops
    if not tf_backbone:
        image_input = tf.keras.Input((None, None, 3))
        backbone_outputs = backbone(image_input)
        x2, x1, x0, _ = backbone_outputs
    else:
        image_input = backbone.inputs
        _ = backbone.get_layer("conv1_relu").output       #/2
        _ = backbone.get_layer("conv2_block3_out").output #/4
        x0 = backbone.get_layer("conv3_block4_out").output #/8
        x1 = backbone.get_layer("conv4_block6_out").output #/16
        x2 = backbone.get_layer("conv5_block3_out").output #/32
        backbone_outputs = x2, x1, x0, _

    if multiscale:
        src_proj_outputs = list((None,None, None, None))
        for i, tensor in enumerate([x0, x1, x2, x2]):
            input_proj_layer = detr.get_layer(f'input_proj/{i}/0')
            input_proj_gn_layer = detr.get_layer(f'input_proj_gn/{i}/1')
            if i == 3:
                tensor = tf.keras.layers.ZeroPadding2D(1)(tensor)
            tensor = input_proj_layer(tensor)

            src_proj_outputs[i] = input_proj_gn_layer(tensor)
    else:
        input_proj_layer = detr.get_layer(f'input_proj/0/0')
        input_proj_gn_layer = detr.get_layer(f'input_proj_gn/0/1')
        src_proj_outputs = [input_proj_gn_layer(input_proj_layer(x2))]

    masks = list(tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1], tf.float32) for x in src_proj_outputs)

    decoder, encoder, outputs_coord = layer_transformer(src_proj_outputs, masks)

    detr = tf.keras.Model(image_input, [outputs_coord, decoder, encoder, src_proj_outputs, backbone_outputs], name="detr_core")

    return detr


def get_deformable_detr_model(
    config,

    include_top=False,
    include_bbox=True,
    nb_class=None,
    weights=None,
    tf_backbone=False,

    batch_size=None,
    num_decoder_layers=6,
    num_encoder_layers=6,

    use_mask_bn=False,


    refine_bbox=False,
    return_intermediate_dec=True,
    model_dim=256,
    multiscale=True,
    include_bbox_3d=False,
    bbox_3d_config=None,
    train_encoder=True,

    ):
    if weights == "deformable-detr-refine_bbox" and nb_class is not None and include_top and nb_class != 91:
        raise ValueError('"deformable_detr" weights are trained with 92 outputs. Do not include the network top to set this number of class')
    elif weights == "deformable_detr" and nb_class is None:
        nb_class = 91

    if weights != "deformable-detr-refine_bbox" and refine_bbox and weights is not None:
        raise ValueError('"Trying to instanciate deformable_detr_bbox_refined with deformable_detr weights')

    init_query_embedding = False #if weights == "deformable_detr" else True
    # Load model and weights
    detr = DeformableDETR(num_classes=nb_class,
                num_decoder_layers=num_decoder_layers,
                num_encoder_layers=num_encoder_layers,
                batch_size=batch_size,
                init_query_embedding=init_query_embedding,
                use_mask_bn=use_mask_bn,

                refine_bbox=refine_bbox,
                return_intermediate_dec=return_intermediate_dec,

                multiscale=multiscale,
                train_encoder=train_encoder)

    image_shape = (None, None, 3)

    # Backbone
    if not tf_backbone:
        backbone = detr.get_layer("backbone")
    else:
        config.normalized_method = "tf_resnet"
        backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(None, None, 3))

    if weights is not None:
        load_weights(detr, weights)

    # Backbone
    if not tf_backbone:
        backbone = detr.get_layer("backbone")
    else:
        backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=image_shape)

    # Get detr core: backbone + transformer
    image_input = tf.keras.Input(image_shape, batch_size=batch_size)

    detr_core_outputs = get_detr_core(detr, backbone, model_dim, tf_backbone=tf_backbone, multiscale=multiscale)(image_input)

    if include_top is False and nb_class is None:
        return tf.keras.Model(image_input, detr_core_outputs, name="detr_core")


    outputs_coord, decoder_state, encoder_state, src_proj_output, backbone_outs = detr_core_outputs

    outputs = {"backbone_outs":list(backbone_outs), "src_proj_output":list(src_proj_output), "encoder_state":encoder_state}

    if include_bbox:

        outputs['bbox_pred_boxes'] = outputs_coord[-1]
        outputs["bbox_aux"] = []
        for i in range(0, outputs_coord.shape[0] - 1):
            outputs["bbox_aux"].append({
                "bbox_pred_boxes": outputs_coord[i]
            })

        # Add bbox head
        class_head = DetrClassHead(detr, include_top=include_top, nb_class=nb_class, refine_bbox=refine_bbox)
        bbox_outputs = class_head(decoder_state)
        config.add_heads([class_head])
        update(outputs, bbox_outputs)

    deformable_detr = tf.keras.Model(image_input, outputs, name="deformable_detr")
    return deformable_detr

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


if __name__ == "__main__":
    main()
