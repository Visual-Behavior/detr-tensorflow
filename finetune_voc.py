""" Example on how to finetune on the VOC dataset
using custom layers
"""

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

from data.voc import load_voc, CLASS_NAME
from networks.detr import get_detr_model
from optimizers import setup_optimizers
from optimizers import gather_gradient, aggregate_grad_and_apply
from logger.training_logging import train_log, valid_log
from loss.loss import get_losses
from inference import numpy_bbox_to_image
from training_config import TrainingConfig, training_config_parser
import training

import wandb
import time


def build_model(config):
    """ Build the model with the pretrained weights
    and add new layers to finetune
    """
    # Input
    image_input = tf.keras.Input((None, None, 3))

    # Load the pretrained model
    detr = get_detr_model(config, include_top=False, weights="detr", num_decoder_layers=6, num_encoder_layers=6)

    # Setup the new layers
    cls_layer = tf.keras.layers.Dense(len(CLASS_NAME))
    pos_layer = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid"),
    ])

    transformer_output = detr(image_input)
    cls_preds = cls_layer(transformer_output)
    pos_preds = pos_layer(transformer_output)

    # Define the main outputs along with the auxialiary loss
    outputs = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}
    outputs["aux"] = [ {"pred_logits": cls_preds[i], "pred_boxes": pos_preds[i]} for i in range(1, 5)]

    detr = tf.keras.Model(image_input, outputs, name="detr_finetuning")
    detr.summary()
    return detr


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset
    train_dt = load_voc("train", config.batch_size, config, augmentation=True)
    valid_dt = load_voc("val", 1, config, augmentation=False)

    # Train/finetune the transformers only
    config.train_backbone = False
    config.train_transformers = True

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)

    # Run the training for 5 epochs
    for epoch_nb in range(5):
        training.eval(detr, valid_dt, config, CLASS_NAME, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb, CLASS_NAME)


if __name__ == "__main__":

    #physical_devices = tf.config.list_physical_devices('GPU')
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    if config.log:
        wandb.init(project="detr-tensorflow", reinit=True)
        
    # Run training
    run_finetuning(config)





