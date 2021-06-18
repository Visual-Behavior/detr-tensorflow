import numpy as np
import os
import cv2
import argparse

from detr_tf import bbox
from detr_tensorrt.TRTExecutor import TRTExecutor
from scipy.special import softmax

def normalized_images(image, normalized_method="torch_resnet"):
    """ Normalized images. torch_resnet is used on finetuning
    since the weights are based on the  original paper training code
    from pytorch. tf_resnet is used when training from scratch with a
    resnet50 traine don tensorflow.
    """
    if normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - channel_avg) / channel_std
        return image.astype(np.float32)
    elif normalized_method == "tf_resnet":
        mean = [103.939, 116.779, 123.68]
        image = image[..., ::-1]
        image = image - mean
        return image.astype(np.float32)
    else:
        raise Exception("Can't handler thid normalized method")

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def get_model_inference(m_outputs: dict, background_class, bbox_format="xy_center", threshold=None):

    #print('get model inference', [key for key in m_outputs])

    # Detr or deformable
    predicted_bbox = m_outputs["pred_boxes"][0] if "pred_boxes" in m_outputs else m_outputs["bbox_pred_boxes"][0]
    predicted_labels = m_outputs["pred_logits"][0] if "pred_logits" in m_outputs else m_outputs["bbox_pred_logits"][0]
    activation = "softmax" if "pred_boxes" in m_outputs else "sigmoid"

    if activation == "softmax": # Detr
        softmax_scores = softmax(predicted_labels, axis=-1)
        predicted_scores = np.max(softmax_scores, axis=-1)
        predicted_labels = np.argmax(softmax_scores, axis=-1)
        bool_filter = predicted_labels != background_class
    else: # Deformable Detr
        sigmoid_scores = sigmoid(predicted_labels)
        predicted_scores = np.max(sigmoid_scores, axis=-1)
        predicted_labels = np.argmax(sigmoid_scores, axis=-1)
        threshold = 0.1 if threshold is None else threshold
        bool_filter = predicted_scores > threshold


    predicted_scores = predicted_scores[bool_filter]
    predicted_labels = predicted_labels[bool_filter]
    predicted_bbox = predicted_bbox[bool_filter]

    if bbox_format == "xy_center":
        predicted_bbox = predicted_bbox
    elif bbox_format == "xyxy":
        predicted_bbox = bbox.xcycwh_to_xy_min_xy_max(predicted_bbox)
    elif bbox_format == "yxyx":
        predicted_bbox = bbox.xcycwh_to_yx_min_yx_max(predicted_bbox)
    else:
        raise NotImplementedError()

    return predicted_bbox, predicted_labels, predicted_scores

def main(engine_path):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))