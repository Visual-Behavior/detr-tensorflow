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
    parser.add_argument("--train_nlayers",  action='store_true',  required=False, default=False, help="Train new layers")

    # How to train
    parser.add_argument("--finetuning",  default=False, required=False, action='store_true', help="Load the model weight before to train")
    parser.add_argument("--batch_size",  type=int, required=False, default=1, help="Batch size to use to train the model")
    parser.add_argument("--gradient_norm_clipping",  type=float, required=False, default=0.1, help="Gradient norm clipping")
    parser.add_argument("--target_batch",  type=int, required=False, default=None, help="When running on a single GPU, aggretate the gradient before to apply.")

    # Learning rate
    parser.add_argument("--backbone_lr",  type=bool, required=False, default=1e-5, help="Train backbone")
    parser.add_argument("--transformers_lr",  type=bool, required=False, default=1e-4, help="Train transformers")
    parser.add_argument("--nlayers_lr",  type=bool, required=False, default=1e-4, help="Train new layers")

    # Logging
    parser.add_argument("--log",  required=False, action="store_true", default=False, help="Log into wandb")

    return parser


class TrainingConfig():

    def __init__(self):

        # Dataset info
        self.data_dir, self.img_dir, self.ann_dir, self.ann_file = None, None, None, None
        self.data = DataConfig(data_dir=None, img_dir=None, ann_file=None, ann_dir=None)
        self.background_class = 0
        self.image_size = 376, 672

        # What to train
        self.train_backbone = False
        self.train_transformers = False
        self.train_nlayers = False

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
        self.nlayers_lr = tf.Variable(1e-4)
        self.nlayers = []

        # Training progress
        self.global_step = 0
        self.log = False

        # Pipeline variables
        self.normalized_method = "torch_resnet"
    
    
    def add_nlayers(self, layers):
        """ Set the new layers to train on the training config
        """
        self.nlayers = [l.name for l in layers]


    def update_from_args(self, args):
        """ Update the training config from args
        """
        args = vars(args)
        for key in args:
            if isinstance(getattr(self, key), tf.Variable):
                getattr(self, key).assign(args[key])
            else:
                setattr(self, key, args[key])
        
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