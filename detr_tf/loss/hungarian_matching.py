from typing import Union,Dict,Tuple
from itertools import product
import tensorflow as tf
import numpy as np

from .. import bbox
from scipy.optimize import linear_sum_assignment


def get_offsets(anchors_xywh, target_bbox_xywh):
    # Return the offset between the boxes in anchors_xywh and the boxes
    # in anchors_xywh

    variances = [0.1, 0.2]

    tiled_a_bbox, tiled_t_bbox = bbox.merge(anchors_xywh, target_bbox_xywh)

    g_cxcy = (tiled_t_bbox[:,:,:2] - tiled_a_bbox[:,:,:2])
    g_cxcy = g_cxcy / (variances[0] * tiled_a_bbox[:,:,2:])

    g_wh = tiled_t_bbox[:,:,2:] / tiled_a_bbox[:,:,2:]
    g_wh = tf.math.log(g_wh) / variances[1]

    return tf.concat([g_cxcy, g_wh], axis=-1)


def np_tf_linear_sum_assignment(matrix):

    indices = linear_sum_assignment(matrix)
    target_indices = indices[0]
    pred_indices = indices[1]

    #print(matrix.shape, target_indices, pred_indices)

    target_selector = np.zeros(matrix.shape[0])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(np.bool)

    pred_selector = np.zeros(matrix.shape[1])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(np.bool)

    #print('target_indices', target_indices)
    #print("pred_indices", pred_indices)

    return [target_indices, pred_indices, target_selector, pred_selector]


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()

    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union



def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def loss_labels(outputs, targets, indices, num_boxes, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], 0,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    empty_weight = torch.ones(81)
    empty_weight[0] = 0.1

    #print("log_softmax(input, 1)", F.softmax(src_logits, 1).mean())
    #print("src_logits", src_logits.shape)
    #print("target_classes", target_classes, target_classes.shape)

    #print("target_classes", target_classes)

    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)
    #print('>loss_ce', loss_ce)
    losses = {'loss_ce': loss_ce}

    #if log:
    #    # TODO this should probably be a separate loss, not hacked in this one here
    #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
    return losses



def loss_boxes(outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    #print("------")
    #assert 'pred_boxes' in outputs
    idx = _get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    #print("target_boxes", target_boxes)
    #print("src_boxes", src_boxes)

    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    #print("loss_bbox", loss_bbox)
    losses = {}
    losses['loss_bbox'] = loss_bbox.sum() / target_boxes.shape[0]
    #print(">loss_bbox", losses['loss_bbox'])

    loss_giou = 1 - torch.diag(generalized_box_iou(
        box_cxcywh_to_xyxy(src_boxes),
        box_cxcywh_to_xyxy(target_boxes)))
    #print('>loss_giou', loss_giou)
    losses['loss_giou'] = loss_giou.sum() / target_boxes.shape[0]
    #print(">loss_giou", losses['loss_giou'])
    return losses

def hungarian_matching(t_bbox, t_class, p_bbox, p_class, fcost_class=1, fcost_bbox=5, fcost_giou=2, slice_preds=True) -> tuple:

    if slice_preds:
        size = tf.cast(t_bbox[0][0], tf.int32)
        t_bbox = tf.slice(t_bbox, [1, 0], [size, 4])
        t_class = tf.slice(t_class, [1, 0], [size, -1])
        t_class = tf.squeeze(t_class, axis=-1)

    # Convert frpm [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    p_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(t_bbox)

    softmax = tf.nn.softmax(p_class)

    # Classification cost for the Hungarian algorithom 
    # On each prediction. We select the prob of the expected class
    cost_class = -tf.gather(softmax, t_class, axis=1)

    # L1 cost for the hungarian algorithm
    _p_bbox, _t_bbox = bbox.merge(p_bbox, t_bbox)
    cost_bbox = tf.norm(_p_bbox - _t_bbox, ord=1, axis=-1)

    # Generalized IOU
    iou, union = bbox.jaccard(p_bbox_xy, t_bbox_xy, return_union=True)
    _p_bbox_xy, _t_bbox_xy = bbox.merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    cost_giou = -(iou - (area - union) / area)

    # Final hungarian cost matrix
    cost_matrix = fcost_bbox * cost_bbox + fcost_class * cost_class + fcost_giou * cost_giou

    selectors = tf.numpy_function(np_tf_linear_sum_assignment, [cost_matrix], [tf.int64, tf.int64, tf.bool, tf.bool] )
    target_indices = selectors[0]
    pred_indices = selectors[1]
    target_selector = selectors[2]
    pred_selector = selectors[3]

    return pred_indices, target_indices, pred_selector, target_selector, t_bbox, t_class
