import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import random

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import tensorflow as tf


def get_size_with_aspect_ratio(w, h, size, max_size=None):
    
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)


def get_multiscale_transform(images,
    scales=[480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
    random_crop=(384, 600),
    random_resize=[400, 500, 600],
    max_size=None):
    """ Coco Augmentation pipeline
    """
    h, w, _ =  images.shape

    one_of = []

    scale = np.random.choice(scales)
    scale_height, scake_width = get_size_with_aspect_ratio(w, h, scale, max_size=max_size)
    one_of.append(
        iaa.Resize({"height": scale_height, "width": scake_width})
    )

    random_resize_crop = []
    if random_resize is not None and len(random_resize) > 0:
        scale = np.random.choice(random_resize)
        resize_height, resize_width = get_size_with_aspect_ratio(w, h, scale)
        random_resize_crop.append(
            iaa.Resize({"height": resize_height, "width": resize_width})
        )
    if random_crop is not None:
        crop_width = random.randint(random_crop[0], random_crop[1])
        crop_height = random.randint(random_crop[0], random_crop[1])
        random_resize_crop.append(
            iaa.CropToFixedSize(crop_width, crop_height)
        )     

    random_resize_crop.append(
        iaa.Resize({"height": scale_height, "width": scake_width})
    )

    one_of.append(iaa.Sequential(random_resize_crop))

    seq = iaa.OneOf(one_of)
    return seq



def get_train_fixedsize_transform(image_size):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        sometimes(iaa.OneOf([
            # Resize complety the image
            iaa.Resize({"width": image_size[1], "height": image_size[0]}, interpolation=ia.ALL),
            # Crop into the image
            iaa.CropToFixedSize(image_size[1], image_size[0]),
            # Affine transform
            iaa.Affine(
                scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, 
            )
        ])),
        # Be sure to resize to the target image size
        iaa.Resize({"width": image_size[1], "height": image_size[0]}, interpolation=ia.ALL)
    ], random_order=False) # apply augmenters in random order
    return seq


def get_valid_fixedsize_transform(image_size):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # Be sure to resize to the target image size
        iaa.Resize({"width": image_size[1], "height": image_size[0]})
    ], random_order=False) # apply augmenters in random order
    return seq


def bbox_xcyc_wh_to_imgaug_bbox(bbox, target_class, height, width):

    img_aug_bbox = []
    
    for b in range(0, len(bbox)):
        bbox_xcyc_wh = bbox[b]
        # Convert size form 0.1 to height/width
        bbox_xcyc_wh = [
            bbox_xcyc_wh[0] * width,
            bbox_xcyc_wh[1] * height,
            bbox_xcyc_wh[2] * width,
            bbox_xcyc_wh[3] * height
        ]
        x1 = bbox_xcyc_wh[0] - (bbox_xcyc_wh[2] / 2)
        x2 = bbox_xcyc_wh[0] + (bbox_xcyc_wh[2] / 2)
        y1 = bbox_xcyc_wh[1] - (bbox_xcyc_wh[3] / 2)
        y2 = bbox_xcyc_wh[1] + (bbox_xcyc_wh[3] / 2)

        n_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=target_class[b])

        img_aug_bbox.append(n_bbox)
    img_aug_bbox
    return img_aug_bbox


def prepare_aug_inputs(image, bbox, t_class):

    images_batch = []
    bbox_batch = []

    images_batch.append(image)

    # Create the Imgaug bbox
    bbs_original = bbox_xcyc_wh_to_imgaug_bbox(bbox, t_class, image.shape[0], image.shape[1])
    bbs_original = BoundingBoxesOnImage(bbs_original, shape=image.shape)
    bbox_batch.append(bbs_original)

    for i in range(len(images_batch)):
        images_batch[i] = images_batch[i].astype(np.uint8)
    
    return images_batch, bbox_batch


def detr_aug_seq(image, config, augmenation):


    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    target_min_side_size = 480

    # According to the paper
    min_side_min = 480
    min_side_max = 800 
    max_side_max = 1333

    image_size = config.image_size
    # Multi scale training
    if image_size is None:
        if augmenation:
            return  get_multiscale_transform(image, max_size=1300)
        else:
            return get_multiscale_transform(
                image, 
                scales=[800],
                random_crop=None,
                random_resize=None,
                max_size=1300
            )
    else:
        if augmenation:
            return get_train_fixedsize_transform(image_size)
        else:
            return get_valid_fixedsize_transform(image_size)

    return seq


def imgaug_bbox_to_xcyc_wh(bbs_aug, height, width):

    bbox_xcyc_wh = []
    t_class = []

    nb_bbox = 0

    for b, bbox in enumerate(bbs_aug):

        h = bbox.y2 - bbox.y1
        w = bbox.x2 - bbox.x1
        xc = bbox.x1 + (w/2)
        yc = bbox.y1 + (h/2)
 
        assert bbox.label != None

        bbox_xcyc_wh.append([xc / width, yc / height, w / width, h / height])
        t_class.append(bbox.label)

        nb_bbox += 1

    #bbox_xcyc_wh[0][0] = nb_bbox
    bbox_xcyc_wh = np.array(bbox_xcyc_wh)

    return bbox_xcyc_wh, t_class


def retrieve_outputs(augmented_images, augmented_bbox):

    outputs_dict = {}
    image_shape = None


    # We expect only one image here for now
    image = augmented_images[0].astype(np.float32)
    augmented_bbox = augmented_bbox[0]

    bbox, t_class = imgaug_bbox_to_xcyc_wh(augmented_bbox, image.shape[0], image.shape[1])

    bbox = np.array(bbox)
    t_class = np.array(t_class)

    return image, bbox, t_class



def detr_transform(image, bbox, t_class, config, augmentation):


    images_batch, bbox_batch = prepare_aug_inputs(image, bbox, t_class)


    seq = detr_aug_seq(image, config, augmentation)

    # Run the pipeline in a deterministic manner
    seq_det = seq.to_deterministic()

    augmented_images = []
    augmented_bbox = []
    augmented_class = []

    for img, bbox, t_cls in zip(images_batch, bbox_batch, t_class):

        img_aug = seq_det.augment_image(img)
        bbox_aug = seq_det.augment_bounding_boxes(bbox)


        for b, bbox_instance in enumerate(bbox_aug.items):
            setattr(bbox_instance, "instance_id", b+1)

        bbox_aug = bbox_aug.remove_out_of_image_fraction(0.7)
        segmap_aug = None
        bbox_aug = bbox_aug.clip_out_of_image()


        augmented_images.append(img_aug)
        augmented_bbox.append(bbox_aug)

    return retrieve_outputs(augmented_images, augmented_bbox)


