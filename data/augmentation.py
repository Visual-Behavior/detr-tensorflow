import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

import tensorflow as tf
# > 2.3
if int(tf.__version__.split('.')[1]) > 3:
    RAGGED = True
else:
    RAGGED = False


def bbox_xcyc_wh_to_imgaug_bbox(bbox, target_class, height, width):
    nb_bbox = bbox[0][0]

    img_aug_bbox = []
    
    if not RAGGED:
        nb_bbox = int(bbox[0][0])
        bbox = bbox[1:nb_bbox+1]
        target_class = target_class[1:nb_bbox+1]

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


def detr_aug_seq():

    SIZE = 540

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # Horizontal flips
        iaa.Fliplr(0.5),
        iaa.OneOf([
            iaa.Resize({"width": SIZE, "height": SIZE}, interpolation=ia.ALL),
            iaa.Sequential([
                iaa.Affine(
                    scale={"x": (0.5, 1.3), "y": (0.5, 1.3)}, 
                ),
                iaa.Resize({"width": SIZE, "height": SIZE}, interpolation=ia.ALL),
            ]),
        ]) 
    ], random_order=False)

    return seq


def imgaug_bbox_to_xcyc_wh(bbs_aug, height, width):

    if RAGGED:
        bbox_xcyc_wh = []
        t_class = []
    else:
        bbox_xcyc_wh = np.zeros((100, 4))
        t_class = np.zeros((100, 1))

    nb_bbox = 0

    for b, bbox in enumerate(bbs_aug):

        h = bbox.y2 - bbox.y1
        w = bbox.x2 - bbox.x1
        xc = bbox.x1 + (w/2)
        yc = bbox.y1 + (h/2)
 
        assert bbox.label != None

        if RAGGED:
            bbox_xcyc_wh.append([xc / width, yc / height, w / width, h / height])
            t_class.append(bbox.label)
        else:
            bbox_xcyc_wh[b+1] = [xc / width, yc / height, w / width, h / height]
            t_class[b+1] = [bbox.label]
        
        nb_bbox += 1

    bbox_xcyc_wh[0][0] = nb_bbox
    bbox_xcyc_wh = np.array(bbox_xcyc_wh)

    return bbox_xcyc_wh, t_class


def retrieve_outputs(augmented_images, augmented_bbox):

    outputs_dict = {}
    image_shape = None


    # We expect only one image here for now
    image = augmented_images[0].astype(np.float32)
    augmented_bbox = augmented_bbox[0]

    bbox, t_class = imgaug_bbox_to_xcyc_wh(augmented_bbox, image.shape[0], image.shape[0])

    return image, bbox, t_class



def detr_aug(image, bbox, t_class):

    # Prepare the augmenation input pipeline
    images_batch, bbox_batch = prepare_aug_inputs(image, bbox, t_class)

    seq = detr_aug_seq()

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


