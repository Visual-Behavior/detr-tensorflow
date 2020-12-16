import tensorflow as tf

def get_backbone(input_shape: tuple, backbone_name, backbone_weights=None, backbone_layers=None, query_prior=None):
    """
        Retrieve a resnet50 or a ResNet101 backbone. Then exteact from the backbone
        the layers used by the FPN.
    """

    with tf.name_scope(backbone_name):
        # Init the backbone
        if backbone_name == "resnet50":
            # Weights are loaded later by the setup_training script or from a checkpoint
            #print("tf.keras.applications.ResNet50", include_backbone_top, backbone_weights, input_shape)
            backbone = tf.keras.applications.ResNet50(include_top=False, weights=backbone_weights, input_shape=tuple(input_shape))         
            #backbone.summary()
            # Selected the needed layers for the FPN
            # conv2_block3_out
            if backbone_layers is None:
                backbone_layers = [
                    backbone.get_layer('conv3_block4_out').output, 
                    backbone.get_layer('conv4_block6_out').output,
                    backbone.get_layer('conv5_block3_out').output,
                ]
            else:
                backbone_layers = [backbone.get_layer(name).output for name in backbone_layers]
            resnet_layers = backbone_layers  
        elif backbone_name == "resnet101":
            # Weights are loaded later by the setup_training script or from a checkpoint
            backbone = tf.keras.applications.ResNet101(include_top=False, weights=backbone_weights, input_shape=tuple(input_shape))
            # Selected the needed layers for the FPN
            if backbone_layers is None:
                backbone_layers = [
                    backbone.get_layer('conv3_block4_out').output, 
                    backbone.get_layer('conv4_block6_out').output,
                    backbone.get_layer('conv5_block3_out').output,
                ]
            else:
                backbone_layers = [backbone.get_layer(name).output for name in backbone_layers]
            resnet_layers = backbone_layers
        elif backbone_name == "mobilenetv2":
            backbone = tf.keras.applications.MobileNetV2(include_top=False, weights=backbone_weights, input_shape=tuple(input_shape))
            #backbone.summary()
            # Selected the needed layers for the FPN
            if backbone_layers is None:
                backbone_layers = [
                    backbone.get_layer('block_5_add').output, 
                    backbone.get_layer('block_7_add').output,
                    backbone.get_layer('block_14_add').output,
                ]
            else:
                backbone_layers = [backbone.get_layer(name).output for name in backbone_layers]
            resnet_layers = backbone_layers     

        elif backbone_name == "efficientnet_b0":
            backbone = tf.keras.applications.EfficientNetB0(include_top=False, weights=backbone_weights, input_shape=tuple(input_shape))
            #backbone.summary()
            # Selected the needed layers for the FPN
            if backbone_layers is None:
                backbone_layers = [
                    backbone.get_layer('block4a_expand_activation').output, 
                    backbone.get_layer('block6a_expand_activation').output,
                    backbone.get_layer('top_activation').output,
                ]
            else:
                backbone_layers = [backbone.get_layer(name).output for name in backbone_layers]
            resnet_layers = backbone_layers
        else:
            raise NotImplementedError(f"Selected backbone_name:{backbone_name} does not exists")

        # Setup the backbone model with the appropriate io
        backbone = tf.keras.Model(inputs=backbone.input, outputs=backbone_layers, trainable=True, name=backbone_name)

        return backbone
