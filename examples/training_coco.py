""" Example on how to train on COCO from scratch
"""


import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

from data.coco import load_coco, CLASS_NAME
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
    """ Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load detr model without weight. 
    # Use the tensorflow backbone with the imagenet weights
    detr = get_detr_model(config, include_top=True, weights=None, tf_backbone=True)
    detr.summary()
    return detr


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset
    train_dt = load_coco("train", config.batch_size, config, augmentation=True)
    valid_dt = load_coco("val", 1, config, augmentation=False)

    # Train the backbone and the transformers
    # Check the training_config file for the other hyperparameters
    config.train_backbone = True
    config.train_transformers = True

    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(detr, config)

    # Run the training for 100 epochs
    for epoch_nb in range(100):
        training.eval(detr, valid_dt, config, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb)


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    if config.log:
        wandb.init(project="detr-tensorflow", reinit=True)
        
    # Run training
    run_finetuning(config)





