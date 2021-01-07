"""
    This scripts is used to send training logs to Wandb.
"""
from typing import Union,Dict,Tuple
import tensorflow as tf
import numpy as np

try:
    # Should be optional
    import wandb
except:
    wandb = None

import cv2

from ..loss.compute_map import cal_map, calc_map, APDataObject

class WandbSender(object):
    """
    Class used within the Yolact project to send data to Wandb to
    log experiments.
    """

    IOU_THRESHOLDS = [x / 100. for x in range(50, 100, 5)]
    AP_DATA = None
    NB_CLASS = None

    def __init__(self):
        self.init_buffer()
    
    @staticmethod
    def init_ap_data(nb_class=None):
        """ Init the ap data used to compute the Map metrics.
            If nb_class is not provided, used the last provided nb_class.
        """
        if nb_class is not None:
            WandbSender.NB_CLASS = nb_class

        if WandbSender.NB_CLASS is None:
            raise ValueError("NB_CLASS is not sed in WandbSender")
        
        if WandbSender.AP_DATA is None:
            WandbSender.AP_DATA = {
                'box' : [[APDataObject() for _ in [f"class_{i}" for i in range(WandbSender.NB_CLASS)]] for _ in [x / 100. for x in range(50, 100, 5)]],
                'mask': [[APDataObject() for _ in [f"class_{i}" for i in range(WandbSender.NB_CLASS)]] for _ in [x / 100. for x in range(50, 100, 5)]]
            }


    def init_buffer(self):
        """ Init list used to store the information from a batch of data.
        Onced the list is filled, the send method
        send all images online.
        """
        self.images = []
        self.queries = []
        self.images_mask_ground_truth = []
        self.images_mask_prediction = []
        self.p_labels_batch = []
        self.t_labels_batch = []
        self.batch_mAP = []


    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def compute_map(p_bbox: np.array, p_labels: np.array, p_scores: np.array, t_bbox: np.array, t_labels: np.array, 
                    b: int, batch: int, prefix: str, step: int, send: bool,
                    p_mask: np.array, t_mask: np.array):
        """
        For some reason, autograph is trying to understand what I'm doing here. With some failure. 
        Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan this method.

        Args:
            p_bbox/t_bbox: List of bbox (n, 4) [y1, x2, y2, x2]
            p_labels/t_labels: List of labels index (n)
            p_mask/t_mask: predicted/target mask (n, h, w) with h and w the size of the mask
            p_scores: List of predicted scores (n)
            b: Batch indice
            batch: size of a batch
            prefix: Prefix to use to log something on wandb
            step: Step number
            send: Whether to send the result of all computed map to wandb.
        """

        # Init Ap Data
        if WandbSender.AP_DATA is None:
            WandbSender.init_ap_data()

        # Set fake class name. (we do not really need the real name of each class at this point)
        class_name = [f"class_{i}" for i in range(WandbSender.NB_CLASS)]

        try:
            # Compyute
            cal_map(p_bbox, p_labels, p_scores, p_mask, t_bbox, t_labels, t_mask, WandbSender.AP_DATA, WandbSender.IOU_THRESHOLDS)

            # Last element of the validation set.

            if send and b + 1 == batch:

                all_maps = calc_map(WandbSender.AP_DATA, WandbSender.IOU_THRESHOLDS, class_name, print_result=True)
                wandb.log({
                    f"val/map50_bbox": all_maps["box"][50],
                    f"val/map50_mask": all_maps["mask"][50],
                    f"val/map_bbox": all_maps["box"]["all"],
                    f"val/map_mask": all_maps["mask"]["all"]
                }, step=step)
                wandb.run.summary.update({
                    f"val/map50_bbox": all_maps["box"][50],
                    f"val/map50_mask": all_maps["mask"][50],
                    f"val/map_bbox": all_maps["box"]["all"],
                    f"val/map_mask": all_maps["mask"]["all"]
                })


                WandbSender.AP_DATA = None
                WandbSender.init_ap_data()

            return np.array([0.0, 0.0], np.float64)

        except Exception as e:
            print("compute_map error. e=", e)
            #raise e
            return np.array([0.0, 0.0], np.float64)
        return np.array([0.0, 0.0], np.float64)


    def get_wandb_bbox_mask_image(self, image: np.array, bbox: np.array, labels : np.array, masks=None, scores=None, class_name=[]) -> Tuple[list, np.array]:
        """
        Serialize the model inference into a dict and an image ready to be send to wandb.
        Args:
            image: (550, 550, 3)
            bbox: List of bbox (n, 4) [x1, y2, x2, y2]
            labels: List of labels index (n)
            masks: predicted/target mask (n, h, w) with h and w the size of the mask
            scores: List of predicted scores (n) (Optional)
            class_name; List of class name for each label
        Return:
           A dict with the box data for wandb
           and a copy of the  original image with the instance masks
        """
        height, width = image.shape[0], image.shape[1]
        image_mask = np.copy(image)
        instance_id = 1
        box_data = []

        for b in range(len(bbox)):
            # Sample a new color for the mask instance
            instance_color = np.random.uniform(0, 1, (3))
            # Retrive bbox coordinates
            x1, y1, x2, y2 = bbox[b]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

            # Fill the mask
            if masks is not None:
                mask = masks[:,:,b]
                mask = cv2.resize(mask, (width, height))
                mask = mask[int(y1*height):int(y2*height),int(x1*width):int(x2*width)]
                image_mask[int(y1*height):int(y2*height),int(x1*width):int(x2*width)][mask > 0.5] = 0.5*image[int(y1*height):int(y2*height),int(x1*width):int(x2*width)][mask > 0.5] + 0.5*instance_color

            image_mask = cv2.rectangle(image_mask, (int(x1*width), int(y1*height)), (int(x2*width), int(y2*height)), (1, 1, 0), 3)


            #if scores is None:
            box_caption = "%s" % (class_name[int(labels[b])])
            #else:
            #    box_caption = "%s-{:.2f}" % (class_name[int(labels[b])], float(scores[b]))

            
            box_dict = {
                "position": {"minX": x1, "maxX": x2, "minY": y1, "maxY": y2},
                "class_id" : int(labels[b]),
                "box_caption" : box_caption
            }
            # b < len(scores) for some reason sometime scores is not of the same length than the bbox
            if scores is not None and b < len(scores):
                box_dict["scores"] = {"conf": float(scores[b])}
            #print("append", box_dict)
            box_data.append(box_dict)
            instance_id += 1        

        return box_data, image_mask

    def gather_inference(self, image: np.array, p_bbox: np.array, p_scores: np.array, t_bbox: np.array, 
                         p_labels: np.array, t_labels: np.array, p_masks=None, t_masks=None, class_name=[]):
        self.class_name = class_name
 
        # This is what wandb expext to get as input to display images with bbox.
        boxes = {
            "ground_truth": {"box_data": []}, 
            "predictions": {"box_data": []}
        }

        # Ground Truth
        box_data, _ = self.get_wandb_bbox_mask_image(image, t_bbox, t_labels, t_masks, class_name=class_name, scores=p_scores)
        boxes["ground_truth"]["box_data"] = box_data
        boxes["ground_truth"]["class_labels"] = {_id:str(label) for _id, label in enumerate(class_name)}

        # Predictions        
        box_data, _ = self.get_wandb_bbox_mask_image(image, p_bbox, p_labels, p_masks, class_name=class_name, scores=p_scores)
        boxes["predictions"]["box_data"] = box_data
        boxes["predictions"]["class_labels"] = {_id:str(label) for _id, label in enumerate(class_name)}


        # Append the target and the predictions to the buffer
        self.images.append(wandb.Image(image, boxes=boxes))

        return np.array(0, dtype=np.int64)
    
    @tf.autograph.experimental.do_not_convert()
    def send(self, step: tf.Tensor, prefix=""):
        """
        For some reason, autograph is trying to understand what I'm doing here. With some failure. 
        Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan this method.

        Send the buffer to wandb
        Args:
            step: The global training step as eager tensor
            prefix: Prefix used before each log name.
        """
        step = int(step)
        
        wandb.log({f"{prefix}Images bbox": self.images}, step=step)
        
        if len(self.batch_mAP) > 0:
            wandb.log({f"{prefix}mAp": np.mean(self.batch_mAP)}, step=step)

        self.init_buffer()

        return np.array(0, dtype=np.int64)

    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def send_depth(depth_map, step: np.array, prefix=""):
        """
        For some reason, autograph is trying to understand what I'm doing here. With some failure. 
        Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan this method.

        Send the depth map to wandb
        Args:
           depth_map: (8, h, w, 1) Depth images used to train the model
           step: The global training step as eager tensor
           prefix: Prefix used before each log name
        """
        step = int(step)
        depth_map_images = []
        for depth in depth_map:
            depth_map_images.append(wandb.Image(depth))
        wandb.log({f"{prefix}Depth map": depth_map_images}, step=step)
        return np.array(0, dtype=np.int64)


    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def send_proto_sample(proto_map: np.array, proto_sample: np.array, proto_targets: np.array, step: np.array, prefix=""):
        """
        For some reason, autograph is trying to understand what I'm doing here. With some failure. 
        Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan this method.

        Send the proto images logs to wandb.
        Args:
           proto_map: The k (32 by default) proto map of the proto network (h, w, k)
           proto_sample: Some generated mask from the network for a batch (n, h, w) with n the number of mask
           proto_targets: The target mask for each generated mask. (n, h, w) with n the number of mask
           step: The global training step as eager tensor
           prefix: Prefix used before each log name
        """
        step = int(step)

        proto_map_images = []
        proto_sample_images = []
        proto_targets_images = []

        for p in range(proto_map.shape[-1]):
            proto_map_images.append(wandb.Image(np.clip(proto_map[:,:,p]*100, 0, 255)))
        for p in range(len(proto_sample)):
            proto_sample_images.append(wandb.Image(proto_sample[p]))
            proto_targets_images.append(wandb.Image(proto_targets[p]))
        
        wandb.log({f"{prefix}Proto Map": proto_map_images}, step=step)
        wandb.log({f"{prefix}Instance segmentation prediction": proto_sample_images}, step=step)
        wandb.log({f"{prefix}Instance segmentation target": proto_targets_images}, step=step)
        return np.array(0, dtype=np.int64)



    @staticmethod
    @tf.autograph.experimental.do_not_convert()
    def send_images(images, step: np.array, name: str, captions=None, masks_prediction=None, masks_target=None):
        """
        For some reason, autograph is trying to understand what I'm doing here. With some failure. 
        Thus, @tf.autograph.experimental.do_not_convert() is used to prevent autograph to scan this method.

        Send some images to wandb
        Args:
           images: (8, h, w, c) Images to log in wandb
           step: The global training step as eager tensor
           name: Image names
        """
        class_labels = {
            0: "background",
            1: "0",
            2: "1",
            3: "2",
            4: "3",
            5: "4",
            6: "5",
            7: "6",
            8: "7",
            9: "8",
            10: "9"
        }

        step = int(step)
        images_list = []
        for i, img in enumerate(images):
            img_params = {}
            if captions is not None:
                img_params["caption"] = captions[i]

            if masks_prediction is not None:
                mask_pred = cv2.resize(masks_prediction[i], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_pred = mask_pred.astype(np.int32)
                if "masks" not in img_params:
                    img_params["masks"] = {}

                #seg = np.expand_dims(masks[i].astype(np.int32), axis=-1)
                img_params["masks"]["predictions"] = {
                    "mask_data": mask_pred,
                    "class_labels": class_labels
                }
                
            
            if masks_target is not None:
                if "masks" not in img_params:
                    img_params["masks"] = {}

                mask_target = masks_target[i].astype(np.int32)
                #seg = np.expand_dims(masks[i].astype(np.int32), axis=-1)
                print(mask_target.shape)
                img_params["masks"]["groud_truth"] = {
                    "mask_data": mask_target,
                     "class_labels": class_labels
                }

            images_list.append(wandb.Image(img, **img_params))

        wandb.log({name: images_list}, step=step)
        return np.array(0, dtype=np.int64)

