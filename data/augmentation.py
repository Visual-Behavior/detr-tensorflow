import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def bbox_xcyc_wh_to_imgaug_bbox(bbox, target_class, height, width):
    nb_bbox = bbox[0][0]
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

        n_bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)#, label=target_class[b+1])

        img_aug_bbox.append(n_bbox)
    img_aug_bbox
    return img_aug_bbox


def prepare_aug_inputs(image, bbox):


    images_batch = []
    bbox_batch = []

    images_batch.append(image)

    augs_io_map = []
    augs_io_map.append({"image": "left_image" , "bbox": None, "target_class": None, "segmap": None})

    augs_io_map[-1]['bbox'] = 'left_bbox'

    target_class = None
    #if "left_categorical_class" in self.io_names and "left_categorical_class" in handlers:
    #    target_class = handlers["left_categorical_class"].output
    #    augs_io_map[-1]["target_class"] = 'left_categorical_class'
    
    # Create the Imgaug bbox
    bbs_original = bbox_xcyc_wh_to_imgaug_bbox(bbox, target_class, image.shape[0], image.shape[1])
    bbs_original = BoundingBoxesOnImage(bbs_original, shape=image.shape)
    bbox_batch.append(bbs_original)

    for i in range(len(images_batch)):
        images_batch[i] = images_batch[i].astype(np.uint8)
    
    return images_batch, bbox_batch


def detr_aug_seq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Resize((540, 540)),
        sometimes(iaa.OneOf([
            iaa.Crop(percent=(0, 0.5)),

            iaa.Affine(
                scale={"x": (0.5, 1.0), "y": (0.5, 1.0)}, 
            )

        ])), 

        iaa.Resize((540, 540))
    ], random_order=False) # apply augmenters in random order

    return seq


def imgaug_bbox_to_xcyc_wh(bbs_aug, height, width):

    bbox_xcyc_wh = []
    padded_target_class = None 
    nb_bbox = 0

    for bbox in bbs_aug:

        h = bbox.y2 - bbox.y1
        w = bbox.x2 - bbox.x1
        xc = bbox.x1 + (w/2)
        yc = bbox.y1 + (h/2)

        bbox_xcyc_wh.append([xc / width, yc / height, w / width, h / height])

    bbox_xcyc_wh = np.array(bbox_xcyc_wh)

    return bbox_xcyc_wh#, padded_target_class


def retrieve_outputs(augmented_images, augmented_bbox):

    outputs_dict = {}
    image_shape = None


    # We expect only one image here for now
    image = augmented_images[0].astype(np.float32)
    augmented_bbox = augmented_bbox[0]

    bbox = imgaug_bbox_to_xcyc_wh(augmented_bbox, image.shape[0], image.shape[0])

    return image, bbox

    """
    for i, io in enumerate(augs_io_map):

        if io["image"] is not None:
            image = augmented_images[i].astype(np.float32)
            outputs_dict[io["image"]] = image
            image_shape = image.shape

        if io["bbox"] is not None:
            padded_bbox, padded_class = self.imgaug_bbox_to_padded_bbox(augmented_bbox[i], image_shape[0], image_shape[1])
            outputs_dict[io["bbox"]] = padded_bbox
            if io["target_class"] is not None and padded_class is not None:
                outputs_dict[io["target_class"]] = padded_class
        
        if io["segmap"] is not None:
            segmap = augmented_segmap[i]
            if segmap is not None:
                outputs_dict[io["segmap"]] = segmap
    """

    return outputs_dict




def detr_aug(image, bbox):

    # Prepare the augmenation input pipeline
    images_batch, bbox_batch = prepare_aug_inputs(image, bbox)
    seq = detr_aug_seq()


    # Run the pipeline in a deterministic manner
    seq_det = seq.to_deterministic()
    augmented_images = []
    augmented_bbox = []

    for img, bbox in zip(images_batch, bbox_batch):
        img_aug = seq_det.augment_image(img)
        bbox_aug = seq_det.augment_bounding_boxes(bbox)

        for b, bbox_instance in enumerate(bbox_aug.items):
            setattr(bbox_instance, "instance_id", b+1)

        bbox_aug = bbox_aug.remove_out_of_image_fraction(0.7)
        segmap_aug = None
        bbox_aug = bbox_aug.clip_out_of_image()

        augmented_images.append(img_aug)
        augmented_bbox.append(bbox_aug)
        #augmented_segmap.append(segmap_aug)

    return retrieve_outputs(augmented_images, augmented_bbox)


