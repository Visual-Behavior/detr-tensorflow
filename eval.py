"""
    Script used to eval a model a Coco for Object detection and segmentation
"""
import argparse
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

from inference import get_model_inference
from data.coco import load_coco

from main import args_to_config
from loss.compute_map import cal_map, calc_map, APDataObject
from networks.detr import get_detr_model
from bbox import xcycwh_to_xy_min_xy_max, xcycwh_to_yx_min_yx_max
from inference import numpy_bbox_to_image

parser = argparse.ArgumentParser()
# Dataset info
parser.add_argument("--cocodir",  type=str, required=True, help="/path/to/coco")
parser.add_argument("--background_class",  type=int, required=False, default=91, help="Default background class")


if int(tf.__version__.split('.')[1]) >= 4:
    RAGGED = True
else:
    RAGGED = False

COCO_CLASS_NAME = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]




def eval_model(config):

    model = get_detr_model(finetuning=True)
    train_dt = load_coco(config.cocodir, "val", batch_size=1, augmentation=False)

    iou_thresholds = [x / 100. for x in range(50, 100, 5)]
    ap_data = {
        'box' : [[APDataObject() for _ in COCO_CLASS_NAME] for _ in iou_thresholds],
        'mask': [[APDataObject() for _ in COCO_CLASS_NAME] for _ in iou_thresholds]
    }



    @tf.function(experimental_relax_shapes=True) #(input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32)])
    def step(images):
        m_outputs = model(images, training=False)
        predicted_bbox, predicted_labels, predicted_scores = get_model_inference(m_outputs, 91, bbox_format="yxyx")
        return predicted_bbox, predicted_labels, predicted_scores


    it = 0
    for images, target_bbox, target_class in train_dt:

        print('images', images.shape)
        continue

        image = numpy_bbox_to_image(images[0].numpy(), bbox_list=target_bbox[0].numpy(), labels=target_class[0].numpy(), class_name=COCO_CLASS_NAME, unnormalized=True)
        plt.imshow(image)
        plt.savefig('test_eval.png')

        p_bbox, p_labels, p_scores = step(images)
        p_mask = None


        t_bbox, t_class = target_bbox[0], target_class[0]
        if not RAGGED:
            size = tf.cast(t_bbox[0][0], tf.int32)
            t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
            t_bbox = xcycwh_to_yx_min_yx_max(t_bbox)
            t_class = tf.slice(t_class, [1, 0], [size, -1])
            t_class = tf.squeeze(t_class, axis=-1)

        # Compute map
        p_mask = np.zeros((138, 138, len(p_bbox)))
        t_mask = np.zeros((138, 138, len(t_bbox)))
        cal_map(p_bbox, p_labels, p_scores, p_mask, np.array(t_bbox), np.array(t_class), t_mask, ap_data, iou_thresholds)

        print(f"Computing map.....{it}", end="\r")
        it += 1
        if it > 200:
            break

    # Compute the mAp
    calc_map(ap_data, iou_thresholds, COCO_CLASS_NAME, print_result=True)

if __name__ == "__main__":
    args = parser.parse_args()
    config = args_to_config(args)

    #parser = argparse.ArgumentParser(description='Eval a model on COCO', add_help=False)
    #parser.add_argument('--limit', type=int, dest='limit')
    #args = parse_scheduler_args(parents=[parser])    
    # Model dir folder
    #model_dir = os.path.join(SURROUND_PATH, "yolobject/models/")
    # Run the evaluation

    eval_model(config)

