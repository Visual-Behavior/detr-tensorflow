from pycocotools.coco import COCO
import tensorflow as tf
import numpy as np
import imageio
from skimage.color import gray2rgb

from data.augmentation import detr_aug #ImageWithLabelsAug


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
    # In the coco dataset the path is split into two part
    # the real path and the id of the associated image (path:id)
    #print("path_info", path_info)

    #path_info = path_info.decode('utf8').split(":")
    #path, img_info = path_info[0], path_info[1]
    #img_id, img_height, img_width = img_info.split("-")
    #bbox = np.zeros((100, 4))
    
    bbox = []
    # Load the labels the instances
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    size = len(anns)
    b = 1

    
    for ann in anns:

        #print("ann", ann)

        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        x_center = bbox_x + (bbox_w / 2)
        y_center = bbox_y + (bbox_h / 2)
        x_center = x_center / float(image_shape[1])
        y_center = y_center / float(image_shape[0])
        bbox_w = bbox_w / float(image_shape[1])
        bbox_h = bbox_h / float(image_shape[0])
        bbox.append([x_center, y_center, bbox_w, bbox_h])

    bbox = np.array(bbox)
    return bbox.astype(np.float32)


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

    bbox = get_coco_labels(coco, coco_id, image.shape)

    if len(bbox) == 0:
        return None, None

    image, bbox = detr_aug(image, bbox)
    if len(bbox) == 0:
        return None, None

    #print("image bbox after aug", image.shape, bbox.shape)
    image = normalized_images(image)

    bbox = np.transpose(bbox, [1, 0])
    bbox = tf.ragged.constant(bbox, dtype=tf.float32)

    return image, bbox

def coco_generator(coco_dir, coco, img_ids, train_val):
    ids = np.random.randint(0, len(img_ids), (len(img_ids),))
    for _id in ids:
        image, bbox = get_coco(coco_dir, coco, img_ids[_id], train_val)
        if image is None:
            continue
        yield image, bbox 



def load_coco(coco_dir, train_val):
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
    
    #def gen():
    #    ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    #    yield 42, ragged_tensor
    #dataset = tf.data.Dataset.from_generator(gen, output_signature=(
    #     tf.TensorSpec(shape=(), dtype=tf.int32),
    #     tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))

    dataset = tf.data.Dataset.from_generator(lambda : coco_generator(coco_dir, coco, img_ids, "val"), output_signature=(
         tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
         tf.RaggedTensorSpec(shape=(4, None), dtype=tf.float32)
    ))
    dataset = dataset.batch(8)

    for images, target_bbox in dataset:
        print(images.shape)
        print(target_bbox.shape)
        print('--')
        #print("v", v.shape, tf.reduce_max(v), tf.reduce_min(v))

    # Setup tensorflow dataset
    #dataset = tf.data.Dataset.from_tensor_slices(imgs_ids)
    #dataset = dataset.map(load_ann)


