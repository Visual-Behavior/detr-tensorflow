import tensorflow as tf
import argparse
import os


def training_config_parser():
    """ Training config class can be overide using the script arguments
    """
    parser = argparse.ArgumentParser()

    # Dataset info
    parser.add_argument("--data_dir",  type=str, required=False, help="Path to the dataset directory")
    parser.add_argument("--img_dir",  type=str, required=False, help="Image directory relative to data_dir")
    parser.add_argument("--ann_file",  type=str, required=False, help="Annotation file relative to data_dir")
    parser.add_argument("--ann_dir",  type=str, required=False, help="Annotation directory relative to data_dir")

    parser.add_argument("--background_class",  type=int, required=False, default=0, help="Default background class")

    # What to train
    parser.add_argument("--train_backbone", action='store_true',  required=False, default=False, help="Train backbone")
    parser.add_argument("--train_transformers", action='store_true',   required=False, default=False, help="Train transformers")
    parser.add_argument("--train_heads",  action='store_true',  required=False, default=False, help="Train the model heads (For finetuning)")

    # How to train
    parser.add_argument("--image_size", default=None, required=False, type=str)
    parser.add_argument("--finetuning",  default=False, required=False, action='store_true', help="Load the model weight before to train")
    parser.add_argument("--batch_size",  type=int, required=False, default=1, help="Batch size to use to train the model")
    parser.add_argument("--gradient_norm_clipping",  type=float, required=False, default=0.1, help="Gradient norm clipping")
    parser.add_argument("--target_batch",  type=int, required=False, default=None, help="When running on a single GPU, aggretate the gradient before to apply.")

    # Learning rate
    parser.add_argument("--backbone_lr",  type=float, required=False, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--transformers_lr",  type=float, required=False, default=1e-4, help="Transformer learning rate")
    parser.add_argument("--heads_lr",  type=float, required=False, default=1e-4, help="Model heads learning rate")

    # Weight decay
    parser.add_argument("--backbone_wd",  type=float, required=False, default=1e-4, help="Backbone weight decay")
    parser.add_argument("--transformers_wd",  type=float, required=False, default=1e-4, help="Transformer weight decay")
    parser.add_argument("--heads_wd",  type=float, required=False, default=1e-4, help="Model heads weight decay")

    # Logging
    parser.add_argument("--log",  required=False, action="store_true", default=False, help="Log into wandb")

    return parser


class TrainingConfig():

    def __init__(self):

        # Dataset info
        self.data_dir, self.img_dir, self.ann_dir, self.ann_file = None, None, None, None
        self.data = DataConfig(data_dir=None, img_dir=None, ann_file=None, ann_dir=None)
        self.background_class = 0

        #self.image_size = 376, 672
        # If image size is None, then multi scale training will be used as 
        # described in the paper.
        self.image_size = None 

        # What to train
        self.train_backbone = False
        self.train_transformers = False
        self.train_heads = False

        # How to train
        self.finetuning = False
        self.batch_size = 1
        self.gradient_norm_clipping = 0.1
        # Batch aggregate before to backprop
        self.target_batch = 1

        # Learning rate
        # Set as tf.Variable so that the variable can be update during the training while
        # keeping the same graph
        self.backbone_lr = tf.Variable(1e-5)
        self.transformers_lr = tf.Variable(1e-4)
        self.heads_lr = tf.Variable(1e-4)

        # Weidht decay
        # Set as tf.Variable so that the variable can be update during the training while
        # keeping the same graph
        self.backbone_wd = tf.Variable(1e-4)
        self.transformers_wd = tf.Variable(1e-4)
        self.heads_wd = tf.Variable(1e-4)

        # Heads layer list
        self.heads = []

        # Training progress
        self.global_step = 0
        self.log = False

        # Pipeline variables
        self.normalized_method = "torch_resnet"
    
    
    def add_heads(self, layers):
        """ Set the new layers to train on the training config
        """
        self.heads = [l.name for l in layers]


    def update_from_args(self, args):
        """ Update the training config from args
        """
        args = vars(args)
        for key in args:
            if isinstance(getattr(self, key, None), tf.Variable):
                getattr(self, key).assign(args[key])
            else:
                setattr(self, key, args[key])
        
        if self.image_size is not None:
            img_size = self.image_size.split(",")
            self.image_size = (int(img_size[0]), int(img_size[1]))

        # Set the config on the data class
        self.data = DataConfig(
            data_dir=self.data_dir,
            img_dir=self.img_dir,
            ann_file=self.ann_file,
            ann_dir=self.ann_dir
        )


class DataConfig():

    def __init__(self, data_dir=None, img_dir=None, ann_file=None, ann_dir=None):
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, img_dir) if data_dir is not None and img_dir is not None else None
        self.ann_file = os.path.join(self.data_dir, ann_file) if ann_file is not None else None
        self.ann_dir = os.path.join(self.data_dir, ann_dir) if ann_dir is not None else None


if __name__ == "__main__":
    args = training_config_parser()
    config = TrainingConfig()
    config.update_from_args(args)