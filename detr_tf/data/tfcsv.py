import tensorflow as tf
from random import shuffle
import pandas as pd
import numpy as np
import imageio
import os

from .import processing
from .transformation import detr_transform
from .. import bbox

def load_data_from_index(index, class_names, filenames, anns, config, augmentation, img_dir):
    # Open the image
    image = imageio.imread(os.path.join(config.data.data_dir, img_dir, filenames[index]))
    # Select all the annotatiom (bbox and class) on this image
    image_anns = anns[anns["filename"] == filenames[index]]    
    
    # Convert all string class to number (the target class)
    t_class = image_anns["class"].map(lambda x: class_names.index(x)).to_numpy()
    # Select the width&height of each image (should be the same since all the ann belongs to the same image)
    width = image_anns["width"].to_numpy()
    height = image_anns["height"].to_numpy()
    # Select the xmin, ymin, xmax and ymax of each bbox, Then, normalized the bbox to be between and 0 and 1
    # Finally, convert the bbox from xmin,ymin,xmax,ymax to x_center,y_center,width,height
    bbox_list = image_anns[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
    bbox_list = bbox_list / [width[0], height[0], width[0], height[0]] 
    t_bbox = bbox.xy_min_xy_max_to_xcycwh(bbox_list)
    
    # Transform and augment image with bbox and class if needed
    image, t_bbox, t_class = detr_transform(image, t_bbox, t_class, config, augmentation=augmentation)

    # Normalized image
    image = processing.normalized_images(image, config)

    return image.astype(np.float32), t_bbox.astype(np.float32), np.expand_dims(t_class, axis=-1).astype(np.int64)


def load_tfcsv_dataset(config, batch_size, augmentation=False, exclude=[], ann_dir=None, ann_file=None, img_dir=None):
    """ Load the hardhat dataset
    """
    ann_dir = config.data.ann_dir if ann_dir is None else ann_dir
    ann_file = config.data.ann_file if ann_file is None else ann_file
    img_dir = config.data.img_dir if img_dir is None else img_dir

    anns = pd.read_csv(os.path.join(config.data.data_dir, ann_file))
    for name  in exclude:
        anns = anns[anns["class"] != name]

    unique_class = anns["class"].unique()
    unique_class.sort()
    

    # Set the background class to 0
    config.background_class = 0
    class_names = ["background"] + unique_class.tolist()


    filenames = anns["filename"].unique().tolist()
    indexes = list(range(0, len(filenames)))
    shuffle(indexes)

    dataset = tf.data.Dataset.from_tensor_slices(indexes)
    dataset = dataset.map(lambda idx: processing.numpy_fc(
        idx, load_data_from_index, 
        class_names=class_names, filenames=filenames, anns=anns, config=config, augmentation=augmentation, img_dir=img_dir)
    ,num_parallel_calls=tf.data.experimental.AUTOTUNE)
    

    # Filter labels to be sure to keep only sample with at least one bbox
    dataset = dataset.filter(lambda imgs, tbbox, tclass: tf.shape(tbbox)[0] > 0)
    # Pad bbox and labels
    dataset = dataset.map(processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Batch images
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    return dataset, class_names

