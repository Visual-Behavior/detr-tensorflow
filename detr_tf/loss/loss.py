import tensorflow as tf
from .. import bbox
from .hungarian_matching import hungarian_matching


def get_total_losss(losses):
    """
    Get model total losss including auxiliary loss
    """
    train_loss = ["label_cost", "giou_loss", "l1_loss"]
    loss_weights = [1, 2, 5]

    total_loss = 0
    for key in losses:
        selector = [l for l, loss_name in enumerate(train_loss) if loss_name in key]
        if len(selector) == 1:
            #print("Add to the total loss", key, losses[key], loss_weights[selector[0]])
            total_loss += losses[key]*loss_weights[selector[0]]
    return total_loss


def get_losses(m_outputs, t_bbox, t_class, config):
    losses = get_detr_losses(m_outputs, t_bbox, t_class, config)

    # Get auxiliary loss for each auxiliary output
    if "aux" in m_outputs:
        for a, aux_m_outputs in enumerate(m_outputs["aux"]):
            aux_losses = get_detr_losses(aux_m_outputs, t_bbox, t_class, config, suffix="_{}".format(a))
            losses.update(aux_losses)
    
    # Compute the total loss
    total_loss = get_total_losss(losses)

    return total_loss, losses


def loss_labels(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector, background_class=0):

    neg_indices = tf.squeeze(tf.where(p_selector == False), axis=-1)
    neg_p_class = tf.gather(p_class, neg_indices)
    neg_t_class = tf.zeros((tf.shape(neg_p_class)[0],), tf.int64) + background_class
    
    neg_weights = tf.zeros((tf.shape(neg_indices)[0],)) + 0.1
    pos_weights = tf.zeros((tf.shape(t_indices)[0],)) + 1.0
    weights = tf.concat([neg_weights, pos_weights], axis=0)
    
    pos_p_class = tf.gather(p_class, p_indices)
    pos_t_class = tf.gather(t_class, t_indices)

    #############
    # Metrics
    #############
    # True negative
    cls_neg_p_class = tf.argmax(neg_p_class, axis=-1)
    true_neg  = tf.reduce_mean(tf.cast(cls_neg_p_class == background_class, tf.float32))
    # True positive
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    true_pos = tf.reduce_mean(tf.cast(cls_pos_p_class != background_class, tf.float32))
    # True accuracy
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    pos_accuracy = tf.reduce_mean(tf.cast(cls_pos_p_class == pos_t_class, tf.float32))

    targets = tf.concat([neg_t_class, pos_t_class], axis=0)
    preds = tf.concat([neg_p_class, pos_p_class], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, preds)
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)

    return loss, true_neg, true_pos, pos_accuracy


def loss_boxes(p_bbox, p_class, t_bbox, t_class, t_indices, p_indices, t_selector, p_selector):
    #print("------")
    p_bbox = tf.gather(p_bbox, p_indices)
    t_bbox = tf.gather(t_bbox, t_indices)


    p_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(p_bbox)
    t_bbox_xy = bbox.xcycwh_to_xy_min_xy_max(t_bbox)

    l1_loss = tf.abs(p_bbox-t_bbox)
    l1_loss = tf.reduce_sum(l1_loss) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    iou, union = bbox.jaccard(p_bbox_xy, t_bbox_xy, return_union=True)

    _p_bbox_xy, _t_bbox_xy = bbox.merge(p_bbox_xy, t_bbox_xy)
    top_left = tf.math.minimum(_p_bbox_xy[:,:,:2], _t_bbox_xy[:,:,:2])
    bottom_right =  tf.math.maximum(_p_bbox_xy[:,:,2:], _t_bbox_xy[:,:,2:])
    size = tf.nn.relu(bottom_right - top_left)
    area = size[:,:,0] * size[:,:,1]
    giou = (iou - (area - union) / area)
    loss_giou = 1 - tf.linalg.diag_part(giou)

    loss_giou = tf.reduce_sum(loss_giou) / tf.cast(tf.shape(p_bbox)[0], tf.float32)

    return loss_giou, l1_loss

def get_detr_losses(m_outputs, target_bbox, target_label, config, suffix=""):

    predicted_bbox = m_outputs["pred_boxes"]
    predicted_label = m_outputs["pred_logits"]

    all_target_bbox = []
    all_target_class = []
    all_predicted_bbox = []
    all_predicted_class = []
    all_target_indices = []
    all_predcted_indices = []
    all_target_selector = []
    all_predcted_selector = []

    t_offset = 0
    p_offset = 0

    for b in range(predicted_bbox.shape[0]):

        p_bbox, p_class, t_bbox, t_class = predicted_bbox[b], predicted_label[b], target_bbox[b], target_label[b]
        t_indices, p_indices, t_selector, p_selector, t_bbox, t_class = hungarian_matching(t_bbox, t_class, p_bbox, p_class, slice_preds=True)

        t_indices = t_indices + tf.cast(t_offset, tf.int64)
        p_indices = p_indices + tf.cast(p_offset, tf.int64)

        all_target_bbox.append(t_bbox)
        all_target_class.append(t_class)
        all_predicted_bbox.append(p_bbox)
        all_predicted_class.append(p_class)
        all_target_indices.append(t_indices)
        all_predcted_indices.append(p_indices)
        all_target_selector.append(t_selector)
        all_predcted_selector.append(p_selector)

        t_offset += tf.shape(t_bbox)[0]
        p_offset += tf.shape(p_bbox)[0]

    all_target_bbox = tf.concat(all_target_bbox, axis=0)
    all_target_class = tf.concat(all_target_class, axis=0)
    all_predicted_bbox = tf.concat(all_predicted_bbox, axis=0)
    all_predicted_class = tf.concat(all_predicted_class, axis=0)
    all_target_indices = tf.concat(all_target_indices, axis=0)
    all_predcted_indices = tf.concat(all_predcted_indices, axis=0)
    all_target_selector = tf.concat(all_target_selector, axis=0)
    all_predcted_selector = tf.concat(all_predcted_selector, axis=0)


    label_cost, true_neg, true_pos, pos_accuracy = loss_labels(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector,
        background_class=config.background_class,
    )

    giou_loss, l1_loss = loss_boxes(
        all_predicted_bbox,
        all_predicted_class,
        all_target_bbox,
        all_target_class,
        all_target_indices,
        all_predcted_indices,
        all_target_selector,
        all_predcted_selector
    )

    label_cost = label_cost
    giou_loss = giou_loss
    l1_loss = l1_loss

    return {
        "label_cost{}".format(suffix): label_cost,
        "true_neg{}".format(suffix): true_neg,
        "true_pos{}".format(suffix): true_pos,
        "pos_accuracy{}".format(suffix): pos_accuracy,
        "giou_loss{}".format(suffix): giou_loss,
        "l1_loss{}".format(suffix): l1_loss
    }
