""" Example on how to finetune on the HardHat dataset
using custom layers. This script assume the dataset is already download 
on your computer in raw and Tensorflow Object detection csv format. 

Please, for more information, checkout the following notebooks:
    - DETR : How to setup a custom dataset
"""

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

from detr_tf.data import load_tfcsv_dataset

from detr_tf.networks.detr import get_detr_model
from detr_tf.optimizers import setup_optimizers
from detr_tf.logger.training_logging import train_log, valid_log
from detr_tf.loss.loss import get_losses
from detr_tf.inference import numpy_bbox_to_image
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf import training

try:
    # Should be optional if --log is not set
    import wandb
except:
    wandb = None

import time


def build_model(config):
    """ Build the model with the pretrained weights
    and add new layers to finetune
    """
    # Load the pretrained model with new heads at the top
    # 3 class : background head and helmet (we exclude here person from the dataset)
    detr = get_detr_model(config, include_top=False, nb_class=3, weights="detr", num_decoder_layers=6, num_encoder_layers=6)
    detr.summary()
    return detr


def run_finetuning(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset and exclude the person class
    train_dt, class_names = load_tfcsv_dataset(
        config, config.batch_size, augmentation=True, exclude=["person"], ann_file="train/_annotations.csv", img_dir="train")
    valid_dt, _ = load_tfcsv_dataset(
        config, 4, augmentation=False, exclude=["person"], ann_file="test/_annotations.csv", img_dir="test")

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

    # Run the training for 180 epochs
    for epoch_nb in range(180):

        if epoch_nb > 0:
            # After the first epoch, we finetune the transformers and the new layers
            config.train_transformers.assign(True)
            config.transformers_lr.assign(1e-4)
            config.nlayers_lr.assign(1e-3)

        training.eval(detr, valid_dt, config, class_names, evaluation_step=100)
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





