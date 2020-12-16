import pickle
import tensorflow as tf
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path


from networks.backbone import get_backbone
from networks.resnet_backbone import ResNet50Backbone
from networks.custom_layers import Linear, FixedEmbedding
from networks.position_embeddings import PositionEmbeddingSine
from networks.transformer import Transformer

from bbox import xcycwh_to_xy_min_xy_max

#from utils import cxcywh2xyxy


def downsample_masks(masks, x):
    masks = tf.cast(masks, tf.int32)
    masks = tf.expand_dims(masks, -1)
    # The existing tf.image.resize with method='nearest'
    # does not expose the half_pixel_centers option in TF 2.2.0
    # The original Pytorch F.interpolate uses it like this
    masks = tf.compat.v1.image.resize_nearest_neighbor(
        masks, tf.shape(x)[1:3], align_corners=False, half_pixel_centers=False)
    masks = tf.squeeze(masks, -1)
    masks = tf.cast(masks, tf.bool)
    return masks

class DETR(tf.keras.Model):
    def __init__(self, num_classes=92, num_queries=100,
                 backbone=None,
                 pos_encoder=None,
                 transformer=None,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 return_intermediate_dec=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_queries = num_queries

        self.backbone = ResNet50Backbone(name='backbone')
        self.transformer = transformer or Transformer(
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                return_intermediate_dec=return_intermediate_dec,
                name='transformer'
        )
        
        self.model_dim = self.transformer.model_dim

        self.pos_encoder = pos_encoder or PositionEmbeddingSine(
            num_pos_features=self.model_dim // 2, normalize=True)

        self.input_proj = tf.keras.layers.Conv2D(self.model_dim, kernel_size=1, name='input_proj')

        self.query_embed = FixedEmbedding((num_queries, self.model_dim),
                                          name='query_embed')

        self.class_embed = Linear(num_classes, name='class_embed')

        self.bbox_embed_linear1 = Linear(self.model_dim, name='bbox_embed_0')
        self.bbox_embed_linear2 = Linear(self.model_dim, name='bbox_embed_1')
        self.bbox_embed_linear3 = Linear(4, name='bbox_embed_2')
        self.activation = tf.keras.layers.ReLU()


    def call(self, inp, training=False, post_process=False):
        x, masks = inp
        x = self.backbone(x, training=training)
        masks = downsample_masks(masks, x)

        pos_encoding = self.pos_encoder(masks)

        hs = self.transformer(self.input_proj(x), masks, self.query_embed(None),
                              pos_encoding, training=training)[0]

        outputs_class = self.class_embed(hs)

        box_ftmps = self.activation(self.bbox_embed_linear1(hs))
        box_ftmps = self.activation(self.bbox_embed_linear2(box_ftmps))
        outputs_coord = tf.sigmoid(self.bbox_embed_linear3(box_ftmps))

        output = {'pred_logits': outputs_class[-1],
                  'pred_boxes': outputs_coord[-1]}

        if post_process:
            output = self.post_process(output)
        return output


    def build(self, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = [(None, None, None, 3), (None, None, None)]
        super().build(input_shape, **kwargs)


def get_detr_model(finetuning=False, nb_class=None, num_decoder_layers=6, num_encoder_layers=6):
    """
    Parameters
    ----------
    finetuning: bool
        If true, the `class_embed` layer will be replace by a new `custom_class_embed` layer.
    nb_class:
        If finetuning is True, `nb_class` will be used for the `custom_class_embed` layer.
    """
    detr = DETR(num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers)

    #detr.summary()
    if finetuning:
        l = detr.load_weights(os.path.join(Path.home(), "weights/detr/dert.ckpt"))
        l.expect_partial()

    image_input = tf.keras.Input((None, None, 3))
    #mask_input = tf.keras.Input((None, None))

    # Backbone
    if finetuning:
        backbone = detr.get_layer("backbone")
    else:
        detr_pt = DETR(num_decoder_layers=num_decoder_layers, num_encoder_layers=num_encoder_layers)
        l = detr_pt.load_weights(os.path.join(Path.home(), "weights/detr/dert.ckpt"))
        l.expect_partial()
        #backbone = get_backbone((None, None, 3), "resnet50", backbone_weights="imagenet", backbone_layers=["conv5_block3_out"])
        backbone = detr_pt.get_layer("backbone")


    # Transformer
    transformer = detr.get_layer("transformer")
    # Positional embedding of the feature map
    position_embedding_sine = detr.get_layer("position_embedding_sine")
    # Used to project the feature map before to fit in into the encoder
    input_proj = detr.get_layer('input_proj')
    # Decoder objects query embedding
    query_embed = detr.get_layer('query_embed')
    
    if nb_class is None:
        # Used to project the  output of the decoder into a class prediction
        # This layer will be replace for finetuning
        class_embed = detr.get_layer('class_embed')
    else:
        class_embed = tf.keras.layers.Dense(nb_class, name="custom_class_embed")

    # Predict the bbox pos
    bbox_embed_linear1 = detr.get_layer('bbox_embed_0')
    bbox_embed_linear2 = detr.get_layer('bbox_embed_1')
    bbox_embed_linear3 = detr.get_layer('bbox_embed_2')
    activation = detr.get_layer("re_lu")


    x = backbone(image_input)
    masks = tf.zeros((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]), tf.bool)  #downsample_masks(mask_input, x)

    pos_encoding = position_embedding_sine(masks)

    hs = transformer(input_proj(x), masks, query_embed(None), pos_encoding)[0]

    outputs_class = class_embed(hs)

    box_ftmps = activation(bbox_embed_linear1(hs))
    box_ftmps = activation(bbox_embed_linear2(box_ftmps))
    outputs_coord = tf.sigmoid(bbox_embed_linear3(box_ftmps))

    outputs = {}

    output = {'pred_logits': outputs_class[-1],
                'pred_boxes': outputs_coord[-1]}

    output["aux"] = []
    for i in range(0, num_decoder_layers - 2):
        out_class = outputs_class[i]
        pred_boxes = outputs_coord[i]
        output["aux"].append({
            "pred_logits": out_class,
            "pred_boxes": pred_boxes
        })


    return tf.keras.Model(image_input, output, name="distill_detr")


def build_mask(image):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    return tf.zeros((h, w), dtype=tf.bool)


def resize(image, min_side=800.0, max_side=1333.0):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cur_min_side = tf.minimum(w, h)
    cur_max_side = tf.maximum(w, h)

    scale = tf.minimum(max_side / cur_max_side,
                       min_side / cur_min_side)
    nh = tf.cast(scale * h, tf.int32)
    nw = tf.cast(scale * w, tf.int32)

    image = tf.image.resize(image, (nh, nw))
    return image


def preprocess_image(image):
    #resized_image = resize(image, min_side=800.0, max_side=1333.0)
    resized_image = resize(image, min_side=550, max_side=550.0)
    channel_avg = tf.constant([0.485, 0.456, 0.406])
    channel_std = tf.constant([0.229, 0.224, 0.225])
    image = (resized_image / 255.0 - channel_avg) / channel_std
    return image, build_mask(image)


@tf.function
def run(model, image, mask):
    return model([image, mask])


if __name__ == "__main__":
    model = get_detr_model(finetuning=True, nb_class=81)



    image = cv2.imread("yoga.jpeg")
    image = image[:,:,::-1]

    img, mask = preprocess_image(image)

    img = tf.expand_dims(img, axis=0)
    mask = tf.expand_dims(mask, axis=0)

    
    for i in range(0, 10):
        t = time.time()
        outputs = run(model, img, mask)
        print("FPS:", 1 / (time.time() - t))

    #model.summary()

    pred_logits = outputs["pred_logits"]
    pred_boxes = outputs["pred_boxes"]

    probs = tf.nn.softmax(pred_logits)
    pred_cls = tf.argmax(probs, axis=-1)
    #print('pred_cls', pred_cls.shape)
    pred_bbox = pred_boxes[pred_cls != 91]

    pred_bbox = xcycwh_to_xy_min_xy_max(pred_bbox)

    image = np.array(image)
    for bbox in pred_bbox:
        x1, y1, x2, y2 = bbox
        image = cv2.rectangle(image, (int(x1*image.shape[1]), int(y1*image.shape[0])), (int(x2*image.shape[1]), int(y2*image.shape[0])), [255, 255, 0], 2)

    plt.imshow(image)
    plt.show()


    

