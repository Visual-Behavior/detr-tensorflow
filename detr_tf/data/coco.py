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

    # If instance into the image, set at least one bbox with -1 everywhere
    # This kind of bbox and class will be ignore at training
    if len(t_bbox) == 0: t_bbox = np.zeros((1, 4)) - 1
    if len(t_class) == 0: t_class = np.zeros((1, 4)) - 1

    # Normalized images
    image = processing.normalized_images(image, config)
    # Set type for tensorflow        
    image = image.astype(np.float32)
    t_bbox = t_bbox.astype(np.float32)
    t_class = t_class.astype(np.int64)
    is_crowd = np.array(is_crowd, dtype=np.int64)

    return image, t_bbox, t_class#, is_crowd


def tensor_to_ragged(image, t_bbox, t_class):
    # Images can have different size in multi-scale training
    # Also, each image can have different number of instance.
    # Therefore, we can use ragged tensor to handle Tensor with dynamic shapes.
    # None is consider as Dynamic in the shape by the Ragged Tensor.
    image.set_shape(tf.TensorShape([None, None, 3]))
    image = tf.RaggedTensor.from_tensor(image).to_tensor()
    t_bbox.set_shape(tf.TensorShape([None, 4]))
    t_bbox = tf.RaggedTensor.from_tensor(t_bbox).to_tensor()
    t_class.set_shape(tf.TensorShape([None, 1]))
    t_class = tf.RaggedTensor.from_tensor(t_class).to_tensor()
    return image, t_bbox, t_class


def iter_tuple_to_dict(data):
    image, t_bbox, t_class = data
    return {
        "images": image,
        "target_bbox": t_bbox,
        "target_class": t_class
    } 


def load_coco_dataset(config, batch_size, augmentation=False, ann_dir=None, ann_file=None, img_dir=None, shuffle=True):
    """ Load a coco dataset

    Parameters
    ----------
    config: TrainingConfig
        Instance of TrainingConfig
    batch_size: int
        Size of the desired batch size
    augmentation: bool
        Apply augmentations on the training data
    ann_dir: str
        Path to the coco dataset
        If None, will be equal to config.data.ann_dir
    ann_file: str
        Path to the ann_file relative to the ann_dir
        If None, will be equal to config.data.ann_file
    img_dir: str
        Path to the img_dir relative to the data_dir
        If None, will be equal to config.data.img_dir
    shuffle : bool
        Shuffle the dataset by default
    """
    ann_dir = config.data.ann_dir if ann_dir is None else ann_dir
    if ann_dir is None:
        ann_file = config.data.ann_file if ann_file is None else os.path.join(config.data_dir, ann_file)
    else:
        ann_file = config.data.ann_file if ann_file is None else os.path.join(ann_dir, ann_file)    
    img_dir = config.data.img_dir if img_dir is None else os.path.join(config.data_dir, img_dir)

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

    if shuffle:
        shuffle(img_ids)
    dataset = tf.data.Dataset.from_tensor_slices(img_ids)
    # Shuffle the dataset
    if shuffle:
        dataset = dataset.shuffle(1000)
    
    # Retrieve img and labels
    outputs_types=(tf.float32, tf.float32, tf.int64)
    dataset = dataset.map(lambda idx: processing.numpy_fc(
        idx, get_coco_from_id, outputs_types=outputs_types, coco=coco, augmentation=augmentation, config=config, img_dir=img_dir)
    , num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    dataset = dataset.map(tensor_to_ragged, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=batch_size, drop_remainder=True))
    dataset = dataset.prefetch(32)

    dataset.itertuple2dict = lambda data: iter_tuple_to_dict(data)
    
    return dataset, class_names