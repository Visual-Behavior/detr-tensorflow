""" Example on how to finetune on the VOC dataset
using custom layers.
"""

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

try:
    # Should be optional if --log is not set
    import wandb
except:
    wandb = None

import os

from detr_tf.data import load_voc_dataset
from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf import training

VOC_CLASS_NAME = [
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

def build_model(config):
    """ Build the model with the pretrained weights
    and add new layers to finetune
    """
    # Input
    image_input = tf.keras.Input((None, None, 3))

    # Load the pretrained model
    detr = get_detr_model(config, include_top=False, weights="detr", num_decoder_layers=6, num_encoder_layers=6)

    # Setup the new layers
    cls_layer = tf.keras.layers.Dense(len(VOC_CLASS_NAME) + 1, name="cls_layer")
    pos_layer = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(4, activation="sigmoid"),
    ], name="pos_layer")
    config.add_nlayers([cls_layer, pos_layer])

    transformer_output = detr(image_input)
    cls_preds = cls_layer(transformer_output)
    pos_preds = pos_layer(transformer_output)

    # Define the main outputs along with the auxialiary loss
    outputs = {'pred_logits': cls_preds[-1], 'pred_boxes': pos_preds[-1]}
    outputs["aux"] = [ {"pred_logits": cls_preds[i], "pred_boxes": pos_preds[i]} for i in range(0, 5)]

    detr = tf.keras.Model(image_input, outputs, name="detr_finetuning")
    detr.summary()
    return detr


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset (for the purpose of this example we're gonna load the training
    # as the validation, but in practise you should have different folder loader for the training and the validation)
    train_dt, class_names = load_voc_dataset(config,  config.batch_size, augmentation=True)
    valid_dt, _ = load_voc_dataset(config, 1, augmentation=False)

    # Train/finetune the transformers only
    config.train_backbone = tf.Variable(False)
    config.train_transformers = tf.Variable(False)
    config.train_nlayers = tf.Variable(True)
    # Learning rate (NOTE: The transformers and the backbone are NOT trained with)
    # a 0 learning rate. They're not trained, but we set the LR to 0 just so that it is clear
    # in the log that both are not trained at the begining
    config.backbone_lr = tf.Variable(0.0)
    config.transformers_lr = tf.Variable(0.0)
    config.nlayers_lr = tf.Variable(1e-3)

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)

    # Run the training for 5 epochs
    for epoch_nb in range(10):

        if epoch_nb > 0:
            # After the first epoch, we finetune the transformers and the new layers
            config.train_transformers.assign(True)
            config.transformers_lr.assign(1e-4)
            config.nlayers_lr.assign(1e-3)

        training.eval(detr, valid_dt, config, class_names, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb, class_names)


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    if config.log:
        wandb.init(project="detr-tensorflow", reinit=True)
        
    # Run training
    run_finetuning(config)





