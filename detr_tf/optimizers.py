import tensorflow as tf
import tensorflow_addons as tfa

def disable_batchnorm_training(model):
    for l in model.layers:
        if hasattr(l, "layers"):
            disable_batchnorm_training(l)
        elif isinstance(l, tf.keras.layers.BatchNormalization):
            l.trainable = False


def get_backbone_trainable_variables(model):
    backbone_variables = []
    # layer [1] is the detr model including the backbone and the transformers

    detr = model.get_layer("detr")
    tr_index = [l.name for l in detr.layers].index('transformer')

    for l, layer in enumerate(detr.layers):
        if l != tr_index:
            backbone_variables += layer.trainable_variables

    return backbone_variables


def get_transformers_trainable_variables(model, exclude=[]):
    transformers_variables = []

    # Transformers variables
    transformers_variables = model.get_layer("detr").get_layer("transformer").trainable_variables

    for layer in model.layers[2:]:
        if layer.name not in exclude:
            transformers_variables += layer.trainable_variables
        else:
            pass

    return transformers_variables


def get_heads_trainables_variables(model, heads_names):
    heads_variables = []
    for nlayer_name in heads_names:
        heads_variables += model.get_layer(nlayer_name).trainable_variables
    return heads_variables


def get_trainable_variables(model, config):

    disable_batchnorm_training(model)

    backbone_variables = []
    transformers_variables = []
    heads_variables = []

    # The gradient will be retrieve for each trainable variable
    backbone_variables = get_backbone_trainable_variables(model)
    transformers_variables = get_transformers_trainable_variables(model, exclude=config.heads)
    heads_variables = get_heads_trainables_variables(model, config.heads)

    return backbone_variables, transformers_variables, heads_variables


def setup_optimizers(model, config):
    """ Method call by the Scheduler to init user data
    """
    @tf.function
    def get_backbone_learning_rate():
        return config.backbone_lr

    @tf.function
    def get_transformers_learning_rate():
        return config.transformers_lr

    @tf.function
    def get_heads_learning_rate():
        return config.heads_lr

    @tf.function
    def get_backbone_wd():
        return config.backbone_lr*config.backbone_wd

    @tf.function
    def get_transformers_wd():
        return config.transformers_lr*config.transformers_wd

    @tf.function
    def get_heads_wd():
        return config.heads_lr*config.heads_wd



    # Disable batch norm on the backbone
    disable_batchnorm_training(model)

    # Optimizers
    backbone_optimizer = tfa.optimizers.AdamW(
        learning_rate=get_backbone_learning_rate, clipnorm=config.gradient_norm_clipping, weight_decay=get_backbone_wd
    )
    transformers_optimizer = tfa.optimizers.AdamW(
        learning_rate=get_transformers_learning_rate, clipnorm=config.gradient_norm_clipping, weight_decay=get_transformers_wd
    )
    heads_optimizer = tfa.optimizers.AdamW(
        learning_rate=get_heads_learning_rate, clipnorm=config.gradient_norm_clipping, weight_decay=get_heads_wd
    )

    # Set trainable variables

    backbone_variables, transformers_variables, heads_variables = [], [], []

    backbone_variables = get_backbone_trainable_variables(model)
    transformers_variables = get_transformers_trainable_variables(model, exclude=config.heads)
    heads_variables = get_heads_trainables_variables(model, config.heads)


    return {
        "backbone_optimizer": backbone_optimizer,
        "transformers_optimizer": transformers_optimizer,
        "heads_optimizer": heads_optimizer,

        "backbone_variables": backbone_variables,
        "transformers_variables": transformers_variables,
        "heads_variables": heads_variables,
    }


def gather_gradient(model, optimizers, total_loss, tape, config, log):

    backbone_variables, transformers_variables, heads_variables = get_trainable_variables(model, config)
    trainables_variables = backbone_variables + transformers_variables + heads_variables

    gradients = tape.gradient(total_loss, trainables_variables)

    # Retrieve the gradients from the tap
    backbone_gradients = gradients[:len(optimizers["backbone_variables"])]
    transformers_gradients = gradients[len(optimizers["backbone_variables"]):len(optimizers["backbone_variables"])+len(optimizers["transformers_variables"])]
    heads_gradients = gradients[len(optimizers["backbone_variables"])+len(optimizers["transformers_variables"]):]

    gradient_steps = {}

    gradient_steps["backbone"] = {"gradients": backbone_gradients}
    gradient_steps["transformers"] = {"gradients": transformers_gradients}
    gradient_steps["heads"] = {"gradients": heads_gradients}

    
    log.update({"backbone_lr": optimizers["backbone_optimizer"]._serialize_hyperparameter("learning_rate")})
    log.update({"transformers_lr": optimizers["transformers_optimizer"]._serialize_hyperparameter("learning_rate")})
    log.update({"heads_lr": optimizers["heads_optimizer"]._serialize_hyperparameter("learning_rate")})

    return gradient_steps



def aggregate_grad_and_apply(name, optimizers, gradients, step, config):

    gradient_aggregate = None
    if config.target_batch is not None:
        gradient_aggregate = int(config.target_batch // config.batch_size)

    gradient_name = "{}_gradients".format(name)
    optimizer_name = "{}_optimizer".format(name)
    variables_name = "{}_variables".format(name)
    train_part_name = "train_{}".format(name)

    if getattr(config, train_part_name):

        # Init the aggregate gradient
        if gradient_aggregate is not None and step % gradient_aggregate == 0:
            optimizers[gradient_name] = [tf.zeros_like(tv) for tv in optimizers[variables_name]]


        if gradient_aggregate is not None:
            # Aggregate the gradient
            optimizers[gradient_name] = [(gradient+n_gradient) if n_gradient is not None else None for gradient, n_gradient in zip(optimizers[gradient_name], gradients) ]
        else:
            optimizers[gradient_name] = gradients

        # Apply gradient if no gradient aggregate or if we finished gathering gradient oversteps
        if gradient_aggregate is None or (step+1) %  gradient_aggregate == 0:
            optimizers[optimizer_name].apply_gradients(zip(optimizers[gradient_name], optimizers[variables_name]))