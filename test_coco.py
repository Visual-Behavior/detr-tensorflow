import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os
#from loss import 

#physical_devices = tf.config.list_physical_devices('GPU')
#for device in physical_devices:
#    tf.config.experimental.set_memory_growth(device, True)

#tf.debugging.set_log_device_placement(True)

from data.coco import load_coco, CLASS_NAME
from networks.detr import get_detr_model
from optimizers import setup_optimizers
from optimizers import gather_gradient, aggregate_grad_and_apply
from logger.training_logging import train_log, valid_log
from loss.loss import get_losses
from inference import numpy_bbox_to_image

parser = argparse.ArgumentParser()

# Dataset info
parser.add_argument("--cocodir",  type=str, required=True, help="/path/to/coco")
parser.add_argument("--background_class",  type=int, required=False, default=91, help="Default background class")

# What to train
parser.add_argument("--train_backbone", action='store_true',  required=False, default=False, help="Train backbone")
parser.add_argument("--train_transformers", action='store_true',   required=False, default=False, help="Train transformers")
parser.add_argument("--train_nlayers",  action='store_true',  required=False, default=False, help="Train new layers")

# How to train
parser.add_argument("--finetuning",  default=False, required=False, action='store_true', help="Load the model weight before to train")
parser.add_argument("--load_backbone",  default=False, required=False, action='store_true', help="Load weights from the backbone only (testing purposes)")
parser.add_argument("--batch_size",  type=int, required=False, default=False, help="Batch size to use to train the model")
parser.add_argument("--gradient_norm_clipping",  type=float, required=False, default=0.1, help="Gradient norm clipping")
parser.add_argument("--target_batch",  type=int, required=False, default=None, help="When running on a single GPU, aggretate the gradient before to apply.")

# Learning rate
parser.add_argument("--backbone_lr",  type=bool, required=False, default=1e-5, help="Train backbone")
parser.add_argument("--transformers_lr",  type=bool, required=False, default=1e-4, help="Train transformers")
parser.add_argument("--nlayers_lr",  type=bool, required=False, default=1e-4, help="Train new layers")

# Logging
parser.add_argument("--log",  required=False, action="store_true", default=False, help="Log into wandb")

import wandb
import time

class Config:
    pass

def args_to_config(args):
    args = vars(args)
    config = Config()
    for key in args:
        setattr(config, key, args[key])
    return config


# 10.809215068817139s for 50x(8, 376, 672, 3)
# 12. for 50x(8, 376, 672, 3)

def run_epoch(train_dt, config):

    i = 0
    t = time.time()
    for images, t_bbox, t_class in train_dt:
    #for data in train_dt:
        #print('data', data)
        #pass
        print(images.shape, t_bbox.shape, t_class.shape)

        #image = numpy_bbox_to_image(np.array(images), bbox_list=t_bbox.numpy(), labels=t_class.numpy(), class_name=CLASS_NAME, unnormalized=True)
        #plt.imshow(image)
        #plt.savefig("test_coco.png")


        i += 1
        if i >= 50:
            print(time.time() - t)
            break
            



def run_training(args):

    train_config = args_to_config(args)
    valid_config = args_to_config(args)
    valid_config.batch_size = 1

    # Load the training dataset
    train_dt = load_coco(args.cocodir, "val", batch_size=train_config.batch_size, augmentation=True)

    run_epoch(train_dt, train_config)


if __name__ == "__main__":
    args = parser.parse_args()
    # Init wandb logging
    wandb_mode = "" if args.log else "dryrun"

    import os
    os.environ["WANDB_MODE"] = wandb_mode
    wandb.init(project="detr-tensorflow", reinit=True)

    # Run training
    run_training(args)





