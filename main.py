import argparse
import tensorflow as tf
#from loss import 

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)


from data.coco import load_coco
from networks.detr import get_detr_model
from optimizers import setup_optimizers
from loss import get_detr_losses, get_losses
from optimizers import gather_gradient, aggregate_grad_and_apply

parser = argparse.ArgumentParser()

# Dataset info
parser.add_argument("--cocodir",  type=str, required=True, help="/path/to/coco")
parser.add_argument("--background_class",  type=int, required=False, default=91, help="Default background class")

# What to train
parser.add_argument("--train_backbone", action='store_true',  required=False, default=False, help="Train backbone")
parser.add_argument("--train_transformers", action='store_true',   required=False, default=False, help="Train transformers")
parser.add_argument("--train_nlayers",  action='store_true',  required=False, default=False, help="Train new layers")


# How to train
parser.add_argument("--batch_size",  type=int, required=False, default=False, help="Batch size to use to train the model")
parser.add_argument("--gradient_norm_clipping",  type=float, required=False, default=0.1, help="Gradient norm clipping")
parser.add_argument("--target_batch",  type=int, required=False, default=None, help="When running on a single GPU, aggretate the gradient before to apply.")

# Learning rate
parser.add_argument("--backbone_lr",  type=bool, required=False, default=1e-5, help="Train backbone")
parser.add_argument("--transformers_lr",  type=bool, required=False, default=1e-4, help="Train transformers")
parser.add_argument("--nlayers_lr",  type=bool, required=False, default=1e-4, help="Train new layers")


def run_epoch(model, train_dt, optimizers, config):

    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)

    for epoch_step, (images, t_bbox, t_class) in enumerate(train_dt):

        with tf.GradientTape() as tape:
            m_outputs = model(images, training=True)
            total_loss, log = get_losses(m_outputs, t_bbox, t_class, config)
            if gradient_aggregate is not None:
                total_loss = total_loss / gradient_aggregate
        
        # Compute gradient for each part of the network
        gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config, log)
        for name in gradient_steps:
            aggregate_grad_and_apply(name, optimizers, gradient_steps[name]["gradients"], epoch_step, config)

def main(args):
    model = get_detr_model(finetuning=True)

    # Load the training dataset
    train_dt = load_coco(args.cocodir, "val", batch_size=args.batch_size)
    # Setup the optimziers and the trainable variables
    optimzers = setup_optimizers(model, args)
    
    run_epoch(model, train_dt, optimzers, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    pass

