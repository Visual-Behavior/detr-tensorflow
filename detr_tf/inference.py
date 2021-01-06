import tensorflow as tf
import numpy as np
import cv2


CLASS_COLOR_MAP = np.random.randint(0, 255, (100, 3))

from detr_tf import bbox

def numpy_bbox_to_image(image, bbox_list, labels=None, scores=None, class_name=[], config=None):
    """ Numpy function used to display the bbox (target or prediction)
    """
    assert(image.dtype == np.float32 and image.dtype == np.float32 and len(image.shape) == 3)

    if config is not None and config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image * channel_std) + channel_avg
        image = (image*255).astype(np.uint8)
    elif config is not None and config.normalized_method == "tf_resnet":
        image = image + mean
        image = image[..., ::-1]
        image = image  / 255
        
    bbox_xcycwh = bbox.np_rescale_bbox_xcycwh(bbox_list, (image.shape[0], image.shape[1])) 
    bbox_x1y1x2y2 = bbox.np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)

    # Set the labels if not defined
    if labels is None: labels = np.zeros((bbox_x1y1x2y2.shape[0]))

    bbox_area = []
    # Go through each bbox
    for b in range(0, bbox_x1y1x2y2.shape[0]):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2-x1)*(y2-y1))

    # Go through each bbox
    for b in np.argsort(bbox_area)[::-1]:
        # Take a new color at reandon for this instance
        instance_color = np.random.randint(0, 255, (3))
        

        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

        # Select the class associated with this bbox
        class_id = labels[int(b)]

        if scores is not None and len(scores) > 0:
            label_name = class_name[int(class_id)]   
            label_name = "%s:%.2f" % (label_name, scores[b])
        else:
            label_name = class_name[int(class_id)]    

        class_color = CLASS_COLOR_MAP[int(class_id)]
    
        color = instance_color
        
        multiplier = image.shape[0] / 500
        cv2.rectangle(image, (x1, y1), (x1 + int(multiplier*15)*len(label_name), y1 + 20), class_color.tolist(), -10)
        cv2.putText(image, label_name, (x1+2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * multiplier, (0, 0, 0), 1)
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)

    return image


def get_model_inference(m_outputs: dict, background_class, bbox_format="xy_center"):

    predicted_bbox = m_outputs["pred_boxes"][0]
    predicted_labels = m_outputs["pred_logits"][0]

    softmax = tf.nn.softmax(predicted_labels)
    predicted_scores = tf.reduce_max(softmax, axis=-1)
    predicted_labels = tf.argmax(softmax, axis=-1)


    indices = tf.where(predicted_labels != background_class)
    indices = tf.squeeze(indices, axis=-1)

    predicted_scores = tf.gather(predicted_scores, indices)
    predicted_labels = tf.gather(predicted_labels, indices)
    predicted_bbox = tf.gather(predicted_bbox, indices)


    if bbox_format == "xy_center":
        predicted_bbox = predicted_bbox
    elif bbox_format == "xyxy":
        predicted_bbox = bbox.xcycwh_to_xy_min_xy_max(predicted_bbox)
    elif bbox_format == "yxyx":
        predicted_bbox = bbox.xcycwh_to_yx_min_yx_max(predicted_bbox)
    else:
        raise NotImplementedError()

    return predicted_bbox, predicted_labels, predicted_scores
