from pycocotools.coco import COCO
import tensorflow as tf
import numpy as np
import imageio
from skimage.color import gray2rgb
from random import sample, shuffle
import os

from . import transformation
from . import processing
import matplotlib.pyplot as plt

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
    'toothbrush', "back"
]

def get_coco_labels(coco, img_id, image_shape, augmentation):
    # Load the labels the instances
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    # Setup bbox
    bbox = []
    t_class = []
    crowd_bbox = 0
    for a, ann in enumerate(anns):
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox'] 
        # target class
        t_cls = ann["category_id"]
        if ann["iscrowd"]:
            crowd_bbox = 1
        # Convert bbox to xc, yc, w, h formast
        x_center = bbox_x + (bbox_w / 2)
        y_center = bbox_y + (bbox_h / 2)
        x_center = x_center / float(image_shape[1])
        y_center = y_center / float(image_shape[0])
        bbox_w = bbox_w / float(image_shape[1])
        bbox_h = bbox_h / float(image_shape[0])
        # Add bbox and class
        bbox.append([x_center, y_center, bbox_w, bbox_h])
        t_class.append([t_cls])
    # Set bbox header
    bbox = np.array(bbox)
    t_class = np.array(t_class)
    return bbox.astype(np.float32), t_class.astype(np.int32), crowd_bbox


def get_coco_from_id(coco_id, coco, augmentation, config, img_dir):
    # Load imag
    img = coco.loadImgs([coco_id])[0]
    # Load image
    #data_type = "train2017" if train_val == "train" else "val2017"
    filne_name = img['file_name']
    image_path = os.path.join(img_dir, filne_name) #f"{config.}/{data_type}/{filne_name}"
    image = imageio.imread(image_path)
    # Graycale to RGB if needed
    if len(image.shape) == 2: image = gray2rgb(image)
    # Retrieve the image label
    t_bbox, t_class, is_crowd = get_coco_labels(coco, img['id'], image.shape, augmentation)
    # Apply augmentations
    if len(t_bbox) > 0 and augmentation is not None:
        image, t_bbox, t_class = transformation.detr_transform(image, t_bbox,  t_class, config, augmentation)
    # Normalized images
    image = processing.normalized_images(image, config)
    # Set type for tensorflow        
    image = image.astype(np.float32)
    t_bbox = t_bbox.astype(np.float32)
    t_class = t_class.astype(np.int64)
    is_crowd = np.array(is_crowd, dtype=np.int64)
    return image, t_bbox, t_class, is_crowd


def load_coco_dataset(config, batch_size, augmentation=False, ann_dir=None, ann_file=None, img_dir=None):
    """ Load a coco dataset
    """
    ann_dir = config.data.ann_dir if ann_dir is None else ann_dir
    ann_file = config.data.ann_file if ann_file is None else ann_file
    img_dir = config.data.img_dir if img_dir is None else img_dir



    coco = COCO(ann_file)

    # Extract CLASS names
    cats = coco.loadCats(coco.getCatIds())
    # Get the max class ID
    max_id = np.array([cat["id"] for cat in cats]).max()
    class_names = ["N/A"] * (max_id + 2) # + 2 for the background class
    # Add the backgrund class at the end
    class_names[-1] = "back"
    config.background_class = max_id + 1
    for cat in cats:
        class_names[cat["id"]] = cat["name"]

    # Setup the data pipeline
    img_ids = coco.getImgIds()
    shuffle(img_ids)
    dataset = tf.data.Dataset.from_tensor_slices(img_ids)
    # Shuffle the dataset
    dataset = dataset.shuffle(1000)
    # Retrieve img and labels
    outputs_types=(tf.float32, tf.float32, tf.int64, tf.int64)
    dataset = dataset.map(lambda idx: processing.numpy_fc(
        idx, get_coco_from_id, outputs_types=outputs_types, coco=coco, augmentation=augmentation, config=config, img_dir=img_dir)
    , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.filter(lambda imgs, tbbox, tclass, iscrowd: tf.shape(tbbox)[0] > 0 and iscrowd != 1)
    dataset = dataset.map(lambda imgs, tbbox, tclass, iscrowd: (imgs, tbbox, tclass), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Pad bbox and labels
    dataset = dataset.map(processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(32)
    
    return dataset, class_names