import tensorflow as tf
import numpy as np
import os
import onnx
import tf2onnx
from pathlib import Path

from detr_tf.training_config import TrainingConfig, training_config_parser

from detr_tf.networks.detr import get_detr_model
from detr_tf.networks.deformable_detr import get_deformable_detr_model

def get_model(config, args):
    if args.model == "detr":
        print("Loading detr...")
        # Load the model with the new layers to finetune
        model = get_detr_model(config, include_top=True, weights="detr")
        config.background_class = 91
        m_output_names = ["pred_logits", "pred_boxes"]
        use_mask = True
        # model.summary()
        # return model
    elif args.model == "deformable-detr":
        print("Loading deformable-detr...")
        model = get_deformable_detr_model(config, include_top=True, weights="deformable-detr")
        m_output_names = ["bbox_pred_logits", "bbox_pred_boxes"]
        # [print(name, model.output[name]) for name in model.output]
        # model.summary()
        use_mask = False
    else:
        raise NotImplementedError()
    # Remove auxliary outputs
    input_image = tf.keras.Input(args.input_shape, batch_size=1, name="input_image")
    if use_mask:
        mask =tf.keras.Input(args.input_shape[:2] + [1], batch_size=1, name="input_mask")
        m_inputs = (input_image, mask)
    else:
        m_inputs = (input_image, )
    all_outputs = model(m_inputs, training=False)
    
    m_outputs = {
        name:tf.identity(all_outputs[name], name=name) 
        for name in m_output_names if name in all_outputs}
    [print(m_outputs[name]) for name in m_outputs]

    model =  tf.keras.Model(m_inputs, m_outputs, name=args.model)
    model.summary() 
    return model


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    parser = training_config_parser()
    parser.add_argument("model", type=str, default="deformable-detr", help="One of 'detr', or 'deformable-detr'")
    parser.add_argument("--input_shape", type=int, default=[1280, 1920], nargs=2, help="ex: 1280 1920 3")
    parser.add_argument('--save_to', type=str, default=None, help="Path to save ONNX file")
    args = parser.parse_args()
    config.update_from_args(args)

    args.input_shape.append(3) # C = 3

    if args.save_to is None:
        args.save_to = os.path.join("weights", args.model, args.model + "_trt")

    # === Load model
    model = get_model(config, args)
    # === Save model to pb file
    if not os.path.isdir(args.save_to):
        os.makedirs(args.save_to)

    # === Save onnx file
    input_spec = [tf.TensorSpec.from_tensor(tensor) for tensor in model.input]
    # print(input_spec)
    output_path = os.path.join(args.save_to, args.model + ".onnx")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=input_spec, 
        opset=13, output_path=output_path)
    print("===== Inputs =======")
    [print(n.name) for n in model_proto.graph.input]
    print("===== Outputs =======")
    [print(n.name) for n in model_proto.graph.output]


    

