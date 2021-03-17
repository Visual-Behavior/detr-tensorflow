import tensorflow as tf
import numpy as np
import cv2

from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf.networks.detr import get_detr_model
from detr_tf.data import processing
from detr_tf.data.coco import COCO_CLASS_NAME
from detr_tf.inference import get_model_inference, numpy_bbox_to_image

@tf.function
def run_inference(model, images, config):
    m_outputs = model(images, training=False)
    predicted_bbox, predicted_labels, predicted_scores = get_model_inference(m_outputs, config.background_class, bbox_format="xy_center")
    return predicted_bbox, predicted_labels, predicted_scores


def run_webcam_inference(detr):

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        # Convert to RGB and process the input image
        model_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_input = processing.normalized_images(model_input, config)

        # Run inference
        predicted_bbox, predicted_labels, predicted_scores = run_inference(detr, np.expand_dims(model_input, axis=0), config)

        frame = frame.astype(np.float32)
        frame = frame / 255
        frame = numpy_bbox_to_image(frame, predicted_bbox, labels=predicted_labels, scores=predicted_scores, class_name=COCO_CLASS_NAME)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    # Load the model with the new layers to finetune
    detr = get_detr_model(config, include_top=True, weights="detr")
    config.background_class = 91

    # Run webcam inference
    run_webcam_inference(detr)
