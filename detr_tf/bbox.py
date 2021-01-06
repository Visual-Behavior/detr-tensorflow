"""
    This file is used to define all the function related to the manipulation
    and comparaison of bbox
"""

from typing import Union,Dict,Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2


def bbox_xcycwh_to_x1y1x2y2(bbox_xcycwh: np.array):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[xc, yc, w, h], ...]
        @img_size (height, width)
    """
    bbox_x1y1x2y2 = np.zeros_like((bbox_xcycwh))
    bbox_x1y1x2y2[:,0] = bbox_xcycwh[:,0] - (bbox_xcycwh[:,2] / 2)
    bbox_x1y1x2y2[:,2] = bbox_xcycwh[:,0] + (bbox_xcycwh[:,2] / 2)
    bbox_x1y1x2y2[:,1] = bbox_xcycwh[:,1] - (bbox_xcycwh[:,3] / 2)
    bbox_x1y1x2y2[:,3] = bbox_xcycwh[:,1] + (bbox_xcycwh[:,3] / 2)
    bbox_x1y1x2y2 = bbox_x1y1x2y2.astype(np.int32)
    return bbox_x1y1x2y2


def intersect(box_a: tf.Tensor, box_b: tf.Tensor) -> tf.Tensor:
    """
    Compute the intersection area between two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The intersection area [a, b] between each bbox. zero if no intersection
    """
    # resize both tensors to [A,B,2] with the tile function to compare
    # each bbox with the anchors:
    # [A,2] -> [A,1,2] -> [A,B,2]
    # [B,2] -> [1,B,2] -> [A,B,2]
    # Then we compute the area of intersect between box_a and box_b.
    # box_a: (tensor) bounding boxes, Shape: [n, A, 4].
    # box_b: (tensor) bounding boxes, Shape: [n, B, 4].
    # Return: (tensor) intersection area, Shape: [n,A,B].

    A = tf.shape(box_a)[0] # Number of possible bbox
    B = tf.shape(box_b)[0] # Number of anchors

    #print(A, B, box_a.shape, box_b.shape)
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymax = tf.tile(tf.expand_dims(box_a[:, 2:], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymax = tf.tile(tf.expand_dims(box_b[:, 2:], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    above_right_corner = tf.math.minimum(tiled_box_a_xymax, tiled_box_b_xymax)


    # Upper Left Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a_xymin = tf.tile(tf.expand_dims(box_a[:, :2], axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b_xymin = tf.tile(tf.expand_dims(box_b[:, :2], axis=0), [A, 1, 1])
    # Select the lower right corner of the intersect area
    upper_left_corner = tf.math.maximum(tiled_box_a_xymin, tiled_box_b_xymin)


    # If there is some intersection, both must be > 0
    inter = tf.nn.relu(above_right_corner - upper_left_corner)
    inter = inter[:, :, 0] * inter[:, :, 1]
    return inter


def jaccard(box_a: tf.Tensor, box_b: tf.Tensor, return_union=False) -> tf.Tensor:
    """
    Compute the IoU of two sets of boxes.
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        The Jaccard overlap [a, b] between each bbox
    """
    # Get the intersectin area
    inter = intersect(box_a, box_b)

    # Compute the A area
    # (xmax - xmin) * (ymax - ymin)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    # Tile the area to match the anchors area
    area_a = tf.tile(tf.expand_dims(area_a, axis=-1), [1, tf.shape(inter)[-1]])

    # Compute the B area
    # (xmax - xmin) * (ymax - ymin)
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    # Tile the area to match the gt areas
    area_b = tf.tile(tf.expand_dims(area_b, axis=-2), [tf.shape(inter)[-2], 1])

    union = area_a + area_b - inter

    if return_union is False:
        # Return the intesect over union
        return inter / union
    else:
        return inter / union, union

def merge(box_a: tf.Tensor, box_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Merged two set of boxes so that operations ca be run to compare them
    Args:
        box_a: A (tf.Tensor) list a bbox (a, 4) with a the number of bbox
        box_b: A (tf.Tensor) list a bbox (b, 4) with b the number of bbox
    Returns:
        Return the two same tensor tiled: (a, b, 4)
    """
    A = tf.shape(box_a)[0] # Number of bbox in box_a
    B = tf.shape(box_b)[0] # Number of bbox in box b
    # Above Right Corner of Intersect Area
    # (b, A, 2) -> (b, A, B, 2)
    tiled_box_a = tf.tile(tf.expand_dims(box_a, axis=1), [1, B, 1])
    # (b, B, 2) -> (b, A, B, 2)
    tiled_box_b = tf.tile(tf.expand_dims(box_b, axis=0), [A, 1, 1])

    return tiled_box_a, tiled_box_b

def xy_min_xy_max_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [ymin, xmin, ymax, xmax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return tf.concat([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)

def yx_min_yx_max_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return tf.concat([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)


def xy_min_xy_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [xc, yc, w, h]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h]
    bbox_xcycwh = tf.concat([bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]], axis=-1)
    return bbox_xcycwh



def xcycwh_to_xy_min_xy_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xyxy = tf.concat([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    # Be sure to keep the values btw 0 and 1
    bbox_xyxy = tf.clip_by_value(bbox_xyxy, 0.0, 1.0)
    return bbox_xyxy


def xcycwh_to_yx_min_yx_max(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [xc, yc, w, h] to [ymin, xmin, ymax, xmax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    bbox = xcycwh_to_xy_min_xy_max(bbox)
    bbox = xy_min_xy_max_to_yx_min_yx_max(bbox)
    return bbox


def yx_min_yx_max_to_xcycwh(bbox: tf.Tensor) -> tf.Tensor:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xc, yc, w, h]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    bbox = yx_min_yx_max_to_xy_min_xy_max(bbox)
    bbox = xy_min_xy_max_to_xcycwh(bbox)
    return bbox



"""
Numpy Transformations
"""

def xy_min_xy_max_to_xcycwh(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xmin, ymin, xmax, ymax] to [xc, yc, w, h]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xmin, ymin, xmax, ymax] to [x_center, y_center, w, h]
    bbox_xcycwh = np.concatenate([bbox[:, :2] + ((bbox[:, 2:] - bbox[:, :2]) / 2), bbox[:, 2:] - bbox[:, :2]], axis=-1)
    return bbox_xcycwh


def np_xcycwh_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xy = np.concatenate([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    return bbox_xy



def np_yx_min_yx_max_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return np.concatenate([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)



def np_rescale_bbox_xcycwh(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[xc, yc, w, h], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_yx_min_yx_max(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[y_min, x_min, y_max, x_max], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_xy_min_xy_max(bbox: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox: [[x_min, y_min, x_max, y_max], ...]
        @img_size (height, width)
    """
    bbox = np.array(bbox) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_rescaled = bbox * scale
    return bbox_rescaled

