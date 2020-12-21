import tensorflow as tf
import xml.etree.ElementTree as ET
from random import sample, shuffle
import numpy as np
import imageio
import numpy as np
import os


from data import processing
from data.augmentation import detr_aug


CLASS_NAME = [
    "back", 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

def load_voc_labels(img_id, voc_dir, augmentation, config):

    anno_path = os.path.join(voc_dir, 'Annotations', img_id + '.xml')
    objects = ET.parse(anno_path).findall('object')
    size = ET.parse(anno_path).find('size')
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    # Set up bbox with headers
    t_bbox = []
    t_class = []

    for obj in objects:
        # Open bbox and retrieve info
        name = obj.find('name').text.lower().strip()
        bndbox = obj.find('bndbox')
        xmin = (float(bndbox.find('xmin').text) - 1) / width
        ymin = (float(bndbox.find('ymin').text) - 1) / height
        xmax = (float(bndbox.find('xmax').text) - 1) / width
        ymax = (float(bndbox.find('ymax').text) - 1) / height
        # Convert bbox to xc, yc center
        xc = xmin + ((xmax - xmin) / 2)
        yc = ymin + ((ymax - ymin) / 2)
        w = xmax - xmin
        h = ymax - ymin
        # Add bbox
        t_bbox.append([xc, yc, w, h])
        # Add target class
        t_class.append([CLASS_NAME.index(name)])

    t_bbox = np.array(t_bbox)
    t_class = np.array(t_class)

    return t_bbox, t_class


def load_voc_from_id(img_id, voc_dir, augmentation, config):
    img_id = str(img_id.decode())
    # Load image
    img_path = os.path.join(voc_dir, 'JPEGImages', img_id + '.jpg')
    image = imageio.imread(img_path)
    # Load labels
    t_bbox, t_class = load_voc_labels(img_id, voc_dir, augmentation, config)
    # Apply augmentations
    if augmentation is not None:
        image, t_bbox, t_class = detr_aug(image, t_bbox,  t_class, augmentation)
    # Normalized images
    image = processing.normalized_images(image, config)
    # Set type for tensorflow        
    image = image.astype(np.float32)
    t_bbox = t_bbox.astype(np.float32)
    t_class = t_class.astype(np.int64)
    return (image, t_bbox, t_class)


def load_voc(train_val, batch_size, config, augmentation=False):
    """
    """
    # Set the background class to 0
    config.background_class = 0

    image_dir = os.path.join(config.datadir, 'JPEGImages')
    anno_dir = os.path.join(config.datadir, 'Annotations')
    ids = list(map(lambda x: x[:-4], os.listdir(image_dir)))
    
    ids = ids[:int(len(ids) * 0.75)] if train_val == "train" else ids[int(len(ids) * 0.75):]
    # Shuffle all the dataset
    shuffle(ids)

    # Setup data pipeline
    dataset = tf.data.Dataset.from_tensor_slices(ids)
    dataset = dataset.shuffle(1000)
    # Retrieve img and labels
    dataset = dataset.map(lambda idx: processing.numpy_fc(idx, load_voc_from_id, voc_dir=config.datadir, augmentation=augmentation, config=config), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Pad bbox and labels
    dataset = dataset.map(processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Filter labels to be sure to keep only sample with at least one bbox  
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tbbox[0][0] > 0)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # Prefetch
    dataset = dataset.prefetch(32)
    return dataset