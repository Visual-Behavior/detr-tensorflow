import tensorflow as tf

import matplotlib.pyplot as plt

from .optimizers import gather_gradient, aggregate_grad_and_apply
from .logger.training_logging import valid_log, train_log
from .loss.loss import get_losses
import time
import wandb


def handle_data(data):
    """ Create (TODO) a mask from the given ragged images. The mask will be use in the encoder/decode 
    attention.
    """
    n_data = {}
    for key in data:
        n_data[key] = data[key]

    padding_mask = tf.ones_like(data["images"])
    # The following will add 0 on all padded part
    padding_mask = padding_mask.to_tensor()[:,:,:,:1]
    # Set one instead of zero on all paded part
    padding_mask = tf.abs(padding_mask - 1)

    n_data["images"] = n_data["images"].to_tensor()
    n_data["mask"] = padding_mask

    return n_data


@tf.function
def run_train_step(model, data, optimizers, config):

    n_data = handle_data(data)

    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)
    else:
        gradient_aggregate = 1

    with tf.GradientTape() as tape:
        m_outputs = model((n_data["images"], n_data["mask"]), training=True)
        total_loss, log = get_losses(m_outputs, t_bbox=n_data["target_bbox"], t_class=n_data["target_class"], config=config, batch_size=config.batch_size)
        total_loss = total_loss / gradient_aggregate

    # Compute gradient for each part of the network
    gradient_steps = gather_gradient(model, optimizers, total_loss, tape, config, log)

    return m_outputs, total_loss, log, gradient_steps


@tf.function
def run_val_step(model, data, config, batch_size):

    n_data = handle_data(data)

    m_outputs = model((n_data["images"], n_data["mask"]), training=False)
    total_loss, log = get_losses(m_outputs, t_bbox=n_data["target_bbox"], t_class=n_data["target_class"], config=config, batch_size=batch_size)
    return m_outputs, total_loss, log


def fit(model, train_dt, optimizers, config, epoch_nb, class_names):
    """ Train the model for one epoch
    """
    # Aggregate the gradient for bigger batch and better convergence
    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)
    t = None
    for epoch_step , data in enumerate(train_dt):

        data = train_dt.itertuple2dict(data)

        # Run the prediction and retrieve the gradient step for each part of the network
        m_outputs, total_loss, log, gradient_steps = run_train_step(model, data, optimizers, config)
        
        # Load the predictions
        if config.log:
            train_log(handle_data(data), m_outputs, config, config.global_step,  class_names, prefix="train/")
        
        # Aggregate and apply the gradient
        for name in gradient_steps:
            aggregate_grad_and_apply(name, optimizers, gradient_steps[name]["gradients"], epoch_step, config)

        # Log every 100 steps
        if epoch_step % 100 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(f"Epoch: [{epoch_nb}], \t Step: [{epoch_step}], \t ce: [{log['label_cost']:.2f}] \t giou : [{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]")
            if config.log:
                wandb.log({f"train/{k}":log[k] for k in log}, step=config.global_step)
            t = time.time()
        
        config.global_step += 1


def eval(model, valid_dt, config, class_name, evaluation_step=200, batch_size=None):
    """ Evaluate the model on the validation set
    """
    batch_size = config.batch_size if batch_size is None else batch_size
    t = None
    for val_step, data in enumerate(valid_dt):
        data = valid_dt.itertuple2dict(data)
        # Run prediction
        m_outputs, total_loss, log = run_val_step(model, data, config, batch_size)
        # Log the predictions
        if config.log:
            valid_log(handle_data(data), m_outputs, config, batch_size, val_step, config.global_step,  class_name, evaluation_step=evaluation_step, prefix="train/")
        # Log the metrics
        if config.log and val_step == 0:
            wandb.log({f"val/{k}":log[k] for k in log}, step=config.global_step)
        # Log the progress
        if val_step % 10 == 0:
            t = t if t is not None else time.time()
            elapsed = time.time() - t
            print(f"Validation step: [{val_step}], \t ce: [{log['label_cost']:.2f}] \t giou : [{log['giou_loss']:.2f}] \t l1 : [{log['l1_loss']:.2f}] \t time : [{elapsed:.2f}]")
        if val_step+1 >= evaluation_step:
            break
