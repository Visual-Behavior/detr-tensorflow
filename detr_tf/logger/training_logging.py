"""
    This script is used to process the results of the training (loss, model outputs and targets)
    In order to send everything on wandb.
"""
from typing import Union,Dict,Tuple
import tensorflow as tf

from ..loss.compute_map import cal_map, calc_map, APDataObject
from .wandb_logging import WandbSender
from ..inference import get_model_inference


import numpy as np
import cv2

from ..import bbox

if int(tf.__version__.split('.')[1]) >= 4:
    RAGGED = True
else:
    RAGGED = False


def tf_send_batch_log_to_wandb(images, target_bbox, target_class,  m_outputs: dict, config, class_name=[], step=None, prefix=""): 

    # Warning: In graph mode, this class is init only once. In eager mode, this class is init at each step.
    img_sender = WandbSender()

    predicted_bbox = m_outputs["pred_boxes"]
    for b in range(predicted_bbox.shape[0]):
        # Select within the batch the elements at indice b
        image = images[b]
        
        elem_m_outputs = {key:m_outputs[key][b:b+1] if (m_outputs[key] is not None and not isinstance(m_outputs[key], list)) else m_outputs[key] for key in m_outputs}

        # Target
        t_bbox, t_class = target_bbox[b], target_class[b]

        if not RAGGED:
            size = tf.cast(t_bbox[0][0], tf.int32)
            t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
            t_bbox = bbox.xcycwh_to_xy_min_xy_max(t_bbox)
            t_class = tf.slice(t_class, [1, 0], [size, -1])
            t_class = tf.squeeze(t_class, axis=-1)

        # Predictions
        predicted_bbox, predicted_labels, predicted_scores = get_model_inference(elem_m_outputs, config.background_class, bbox_format="xyxy")

        np_func_params = {
            "image": image, "p_bbox": np.array(predicted_bbox), "p_scores": np.array(predicted_scores), "t_bbox": np.array(t_bbox),
            "p_labels": np.array(predicted_labels), "t_labels": np.array(t_class), "class_name": class_name
        }
        img_sender.gather_inference(**np_func_params)

    img_sender.send(step=step, prefix=prefix)



def compute_map_on_batch(images, target_bbox, target_class,  m_outputs: dict, config, class_name=[], step=None, send=True, prefix=""): 
    predicted_bbox = m_outputs["pred_boxes"]
    batch_size = predicted_bbox.shape[0]
    for b in range(batch_size):

        image = images[b]
        elem_m_outputs = {key:m_outputs[key][b:b+1] if (m_outputs[key] is not None and not isinstance(m_outputs[key], list)) else m_outputs[key] for key in m_outputs}

        # Target
        t_bbox, t_class = target_bbox[b], target_class[b]

        if not RAGGED:
            size = tf.cast(t_bbox[0][0], tf.int32)
            t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
            t_bbox = bbox.xcycwh_to_yx_min_yx_max(t_bbox)
            t_class = tf.slice(t_class, [1, 0], [size, -1])
            t_class = tf.squeeze(t_class, axis=-1)

        # Inference ops
        predicted_bbox, predicted_labels, predicted_scores = get_model_inference(elem_m_outputs, config.background_class, bbox_format="yxyx")
        pred_mask = None
  
        pred_mask = np.zeros((138, 138, len(predicted_bbox)))
        target_mask = np.zeros((138, 138, len(t_bbox)))
        WandbSender.compute_map(
            np.array(predicted_bbox), 
            np.array(predicted_labels), np.array(predicted_scores), 
            np.array(t_bbox), 
            np.array(t_class), 
            b, batch_size, prefix, step, send, pred_mask, target_mask)



def train_log(images, t_bbox, t_class, m_outputs: dict, config, step, class_name=[], prefix="train/"):
    # Every 1000 steps, log some progress of the training
    # (Images with bbox and images logs)
    if step % 100 == 0:
        tf_send_batch_log_to_wandb(images, t_bbox, t_class, m_outputs, config, class_name=class_name, step=step, prefix=prefix)


def valid_log(images, t_bbox, t_class, m_outputs: dict, config, step, global_step, class_name=[], evaluation_step=200, prefix="train/"):

    # Set the number of class
    WandbSender.init_ap_data(nb_class=len(class_name))
    map_list = compute_map_on_batch(images, t_bbox, t_class, m_outputs, config, class_name=class_name, step=global_step, send=(step+1==evaluation_step),  prefix="val/")
    
    if step == 0:
        tf_send_batch_log_to_wandb(images, t_bbox, t_class, m_outputs, config, class_name=class_name, step=global_step, prefix="val/")
