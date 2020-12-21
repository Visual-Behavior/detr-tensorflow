""" Eval a model on the coco dataset
"""

import argparse
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from inference import get_model_inference
from data.coco import load_coco, CLASS_NAME

from main import args_to_config
from loss.compute_map import cal_map, calc_map, APDataObject
from networks.detr import get_detr_model
from bbox import xcycwh_to_xy_min_xy_max, xcycwh_to_yx_min_yx_max
from inference import numpy_bbox_to_image
from training_config import TrainingConfig, training_config_parser


def build_model(config):
    """ Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load the pretrained model
    detr = get_detr_model(include_top=True, weights="detr")
    detr.summary()
    return detr


def eval_model(model, config, class_names, valid_dt):
    """ Run evaluation
    """

    iou_thresholds = [x / 100. for x in range(50, 100, 5)]
    ap_data = {
        'box' : [[APDataObject() for _ in class_names] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in class_names] for _ in iou_thresholds]
    }

    it = 0
    for images, target_bbox, target_class in valid_dt:
        # Forward pass
        m_outputs = model(images)
        # Run predictions
        p_bbox, p_labels, p_scores = get_model_inference(m_outputs, config.background_class, bbox_format="yxyx")
        # Remove padding
        t_bbox, t_class = target_bbox[0], target_class[0]
        size = tf.cast(t_bbox[0][0], tf.int32)
        t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
        t_bbox = xcycwh_to_yx_min_yx_max(t_bbox)
        t_class = tf.slice(t_class, [1, 0], [size, -1])
        t_class = tf.squeeze(t_class, axis=-1)
        # Compute map
        cal_map(p_bbox, p_labels, p_scores,  np.zeros((138, 138, len(p_bbox))), np.array(t_bbox), np.array(t_class), np.zeros((138, 138, len(t_bbox))), ap_data, iou_thresholds)
        print(f"Computing map.....{it}", end="\r")
        it += 1
        #if it > 100:
        #    break

    # Compute the mAp over all thresholds
    calc_map(ap_data, iou_thresholds, class_names, print_result=True)

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    # Load the model with the new layers to finetune
    detr = build_model(config)

    valid_dt = load_coco("val", 1, config, augmentation=None)

    # Run training
    eval_model(detr, config, CLASS_NAME, valid_dt)


