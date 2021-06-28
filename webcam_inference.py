import tensorflow as tf
import numpy as np
import cv2

from detr_tf.training_config import TrainingConfig, training_config_parser

from detr_tf.networks.detr import get_detr_model
from detr_tf.networks.deformable_detr import get_deformable_detr_model

from detr_tf.data import processing
from detr_tf.data.coco import COCO_CLASS_NAME
from detr_tf.inference import get_model_inference, numpy_bbox_to_image


@tf.function
def run_inference(model, images, config, use_mask=True):

    if use_mask:
        mask = tf.zeros((1, images.shape[1], images.shape[2], 1))
        m_outputs = model((images, mask), training=False)
    else:
        m_outputs = model(images, training=False)

    predicted_bbox, predicted_labels, predicted_scores = get_model_inference(m_outputs, config.background_class, bbox_format="xy_center", threshold=0.2)
    return predicted_bbox, predicted_labels, predicted_scores


def run_webcam_inference(model, use_mask=True):

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        # Convert to RGB and process the input image
        model_input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        model_input = processing.normalized_images(model_input, config)

        # Run inference
        predicted_bbox, predicted_labels, predicted_scores = run_inference(model, np.expand_dims(model_input, axis=0), config, use_mask=use_mask)

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
    parser = training_config_parser()
    
    # Logging
    parser.add_argument("model", type=str, help="One of 'detr', or 'deformable-detr'")
    args = parser.parse_args()
    config.update_from_args(args)

    if args.model == "detr":
        print("Loading detr...")
        # Load the model with the new layers to finetune
        model = get_detr_model(config, include_top=True, weights="detr")
        config.background_class = 91
        use_mask = True
    elif args.model == "deformable-detr":
        print("Loading deformable-detr...")
        model = get_deformable_detr_model(config, include_top=True, weights="deformable-detr")
        model.summary()
        use_mask = False

    # Run webcam inference
    run_webcam_inference(model, use_mask=use_mask)
