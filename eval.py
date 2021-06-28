""" Eval a model on the coco dataset
"""

import argparse
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from detr_tf.inference import get_model_inference
from detr_tf.data.coco import load_coco_dataset
from detr_tf.loss.compute_map import cal_map, calc_map, APDataObject
from detr_tf.networks.detr import get_detr_model
from detr_tf.bbox import xcycwh_to_xy_min_xy_max, xcycwh_to_yx_min_yx_max
from detr_tf.inference import numpy_bbox_to_image
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf.training import handle_data

tf.random.set_seed(40)
np.random.seed(40)



def build_model(config):
    """ Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load the pretrained model
    detr = get_detr_model(config, include_top=True, weights="detr")
    detr.summary()
    return detr


#@tf.function
def run_model(data, model):
    n_data = handle_data(data)
    return model((n_data["images"], n_data["mask"]))


def eval_model(model, config, class_names, valid_dt):
    """ Run evaluation
    """

    iou_thresholds = [x / 100. for x in range(50, 100, 5)]
    ap_data = {
        'box' : [[APDataObject() for _ in class_names] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in class_names] for _ in iou_thresholds]
    }
    it = 0

    for data in valid_dt:
        data = valid_dt.itertuple2dict(data)

        # Forward pass
        m_outputs = run_model(data, model)
        
        # Run predictions
        p_bbox, p_labels, p_scores = get_model_inference(m_outputs, config.background_class, bbox_format="yxyx")
        
        # Remove padding
        t_bbox, t_class = data["target_bbox"][0], data["target_class"][0]

        t_bbox = xcycwh_to_yx_min_yx_max(t_bbox)
        t_class = tf.squeeze(t_class, axis=-1)

        # Filter undesired target
        _filter = tf.squeeze(tf.where(t_class != -1), axis=-1)
        t_class = tf.gather(t_class, _filter)
        t_bbox = tf.gather(t_bbox, _filter)
        
        # Compute map
        cal_map(p_bbox, p_labels, p_scores,  np.zeros((138, 138, len(p_bbox))), np.array(t_bbox), np.array(t_class), np.zeros((138, 138, len(t_bbox))), ap_data, iou_thresholds)
        print(f"Computing map.....{it}", end="\r")
        it += 1
        #if it > 10:
        #    break

    # Compute the mAp over all thresholds
    calc_map(ap_data, iou_thresholds, class_names, print_result=True)

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    # Load the model with the new layers to finetune
    detr = build_model(config)

    valid_dt, class_names = load_coco_dataset(config, 1, augmentation=False, shuffle_data=False)

    # Run training
    eval_model(detr, config, class_names, valid_dt)


