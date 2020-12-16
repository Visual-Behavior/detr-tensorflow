from pycocotools.coco import COCO
import tensorflow as tf
import numpy as np
import imageio
from skimage.color import gray2rgb
import os

from inference import numpy_bbox_to_image
from data.augmentation import detr_aug #ImageWithLabelsAug
import matplotlib.pyplot as plt


# > 2.3
if int(tf.__version__.split('.')[1]) > 3:
    RAGGED = True
else:
    RAGGED = False

CLASS_NAME = [
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

def get_catid_to_info(coco):
    # Get the list of all categories
    cats = coco.loadCats(coco.getCatIds())
    cats_ids_to_info = {}
    nb = 1
    for cat in cats:
        cat["softmax_class"] = cat["id"]
        cats_ids_to_info[cat["id"]] = cat
        nb += 1
    return cats_ids_to_info


def normalized_images(image):
    channel_avg = np.array([0.485, 0.456, 0.406])
    channel_std = np.array([0.229, 0.224, 0.225])
    image = (image / 255.0 - channel_avg) / channel_std
    return image


def get_coco_labels(coco, img_id, image_shape):
    
    bbox = []
    t_class = []

    # Load the labels the instances
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    size = len(anns)
    b = 1

    if not RAGGED:
        bbox = np.zeros((100, 4))
        t_class = np.zeros((100, 1))

    nb_bbox = 0
    for a, ann in enumerate(anns):
        # Target bbox
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox'] 
        # target class
        t_cls = ann["category_id"]

        x_center = bbox_x + (bbox_w / 2)
        y_center = bbox_y + (bbox_h / 2)
        x_center = x_center / float(image_shape[1])
        y_center = y_center / float(image_shape[0])
        bbox_w = bbox_w / float(image_shape[1])
        bbox_h = bbox_h / float(image_shape[0])
        
        if RAGGED:            
            bbox.append([x_center, y_center, bbox_w, bbox_h])
            t_class.append([t_cls])
        else:
            bbox[a+1] = [x_center, y_center, bbox_w, bbox_h]
            t_class[a+1][0] = t_cls
        nb_bbox += 1

    if not RAGGED:
        bbox[0][0] = nb_bbox # A is the number of bbox

    bbox = np.array(bbox)
    t_class = np.array(t_class)

    return bbox.astype(np.float32), t_class.astype(np.int32)


def get_coco(coco_dir, coco, coco_id, train_val):

    img = coco.loadImgs(coco_id)[0]

    if train_val == "train":
        data_type = "train2017"
    elif train_val == "val":
        data_type = "val2017"

    filne_name = img['file_name']
    image_path = f"{coco_dir}/{data_type}/{filne_name}"

    image = imageio.imread(image_path)

    # Graycale
    if len(image.shape) == 2:
        image = gray2rgb(image)

    t_bbox, t_class = get_coco_labels(coco, coco_id, image.shape)

    if (not RAGGED and t_bbox[0][0] == 0) or (RAGGED and len(t_bbox) == 0):
        return None, None, None

    image, t_bbox, t_class = detr_aug(image, t_bbox,  t_class)
    if (not RAGGED and t_bbox[0][0] == 0) or (RAGGED and len(t_bbox) == 0):
        return None, None, None

    image = normalized_images(image)

    if RAGGED:
        t_bbox = tf.ragged.constant(t_bbox, dtype=tf.float32)
        t_class = tf.ragged.constant(t_class, dtype=tf.int64)

    return image, t_bbox, t_class

def coco_generator(coco_dir, coco, img_ids, train_val):
    ids = np.random.randint(0, len(img_ids), (len(img_ids),))
    for _id in ids:
        image, bbox, t_class = get_coco(coco_dir, coco, img_ids[_id], train_val)
        if image is None:
            continue
        yield image, bbox, t_class



def load_coco(coco_dir, train_val, batch_size):
    """
    """
    if train_val == "train":
        data_type = "train2017"
    elif train_val == "val":
        data_type = "val2017"
    # Annotafion file
    ann_file = f"{coco_dir}/annotations/instances_{data_type}.json"
    coco = COCO(ann_file)

    cats_ids_to_info = get_catid_to_info(coco)
    nb_category = np.max([cats_ids_to_info[key]["softmax_class"] for key in cats_ids_to_info]) + 2

    img_ids = coco.getImgIds()
    
    if RAGGED:
        params = {"output_signature": (
            tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int64)
        )}
    else:
        params =  {"output_types": (
                tf.float32,
                tf.float32,
                tf.int64
            ),
            "output_shapes": (
                tf.TensorShape([None, None, 3]),
                tf.TensorShape([100, 4]),
                tf.TensorShape([100, 1])
        )}

    dataset = tf.data.Dataset.from_generator(lambda : coco_generator(coco_dir, coco, img_ids, "val"), **params)
    dataset = dataset.batch(batch_size)

    return dataset


if __name__ == "__main__":
    import argparse
    from epoch import run_epoch
    parser = argparse.ArgumentParser()
    parser.add_argument("--cocodir",  type=str, required=True, help="/path/to/coco")
    args = parser.parse_args()
    train_dt = load_coco(args.cocodir, "val", batch_size=1)

    for images, target_bbox, target_class in train_dt:
        print(images.shape)
        print(target_bbox.shape)
        print(target_class.shape)
        print('--')
        image = numpy_bbox_to_image(np.array(images[0]), bbox_list=target_bbox[0].numpy(), labels=target_class[0].numpy(), class_name=CLASS_NAME, unnormalized=True)

        plt.imshow(image)
        plt.show()

