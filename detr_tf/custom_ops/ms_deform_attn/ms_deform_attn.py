import tensorflow as tf

#Python failback of MSDeformAttnFunction

def MSDeformAttnFunction(values, sampling_locations, attention_weights):

    # for debug and test only,
    # need to use cuda version instead
    """
    :param values                        level, (N, H, W, num_heads, head_dim)
    :param sampling_locations            level, (N, Len_q, num_heads, num_sampling_points, 2)
    :param attention_weights              N, Len_q, num_heads, num_level, num_sampling_points
    """

    sampling_value_list = []
    for lid_, (value, sl) in enumerate(zip(values, sampling_locations)):
        N, h_l, w_l, num_heads, head_dim = tf.unstack(tf.shape(value))
        # N*num_heads, h, w, c
        value = tf.reshape(tf.transpose(value, [0, 3, 1, 2, 4]), [N*num_heads, h_l, w_l, head_dim])

        # N, Len_q, num_heads, num_sampling_points, 2
        sl = 2 * sl - 1 #between (-1, 1)
        N, Len_q, num_heads, num_sampling_points, _ = tf.unstack(tf.shape(sl))

        # N*num_heads, Len_q, num_sampling_points, 2
        sampling_grid_l_ = tf.reshape(tf.transpose(sl, [0, 2, 1, 3, 4]), [N*num_heads, Len_q, num_sampling_points, 2])

        #N*num_heads, Len_q, num_sampling_points, c
        if True:
            sampled_values = bilinear_sampler(value, sampling_grid_l_)
        else:
            sampled_values = nearest_sampler(value, sampling_grid_l_)

        sampling_value_list.append(sampled_values)

    # N*num_heads, Len_q, num_level, num_sampling_points, c
    sampling_value = tf.stack(sampling_value_list, axis=2)
    # N, num_heads, Len_q, num_level, num_sampling_points, c
    sampling_value = tf.reshape(sampling_value, (N, num_heads, Len_q, len(values), num_sampling_points, head_dim))
    # N, Len_q, num_heads, num_level, num_sampling_points, c
    sampling_value = tf.transpose(sampling_value, [0, 2, 1, 3, 4, 5])
    # (N, Len_q, num_heads, num_level, num_sampling_points, 1)
    attention_weights = tf.expand_dims(attention_weights, -1)
    # N, Len_q, num_heads, num_level, num_sampling_points, c
    output = attention_weights * sampling_value
    # N, Len_q, num_heads, -1, head_dim
    output = tf.reshape(output, (N, Len_q, num_heads, -1, head_dim))
    # N, Len_q, num_heads, c
    output = tf.reduce_sum(output, axis=3)

    output = tf.reshape(output, (N, Len_q, num_heads*head_dim))

    return output


def within_bounds(x, lower, upper):
    lower_tensor = tf.greater_equal(x, lower)
    upper_tensor = tf.less_equal(x, upper)
    return tf.logical_and(lower_tensor, upper_tensor)

def bilinear_sampler(image, coords):
    ''' Value sampler using tf.gather_nd
    Args:
      image: tensor with shape (bs, h, w, c)
      coords: coordinates tensor with shape (bs, ... , 2), xy-indexing between 0, 1

    Returns:
      sampled tensor with shape (bs, ... , c)
    '''

    #Correspond to padding="zeros" (optimistic : discard only out of bound bilinear coefficient, not the full value)

    with tf.name_scope("bilinear_sampler"):
      _, h, w, _ = tf.unstack(tf.shape(image))


      gx, gy = tf.unstack(coords, axis=-1)

      # rescale x and y to [0, W-1/H-1]
      gx = (gx+1.0)/2.0  * tf.cast(w-1, tf.float32)
      gy = (gy+1.0)/2.0  * tf.cast(h-1, tf.float32)

      gx0 = tf.floor(gx)
      gx1 = gx0 + 1.0
      gy0 = tf.floor(gy)
      gy1 = gy0 + 1.0

      mx0 = within_bounds(gx0, 0, tf.cast(w, tf.float32)-1)
      mx1 = within_bounds(gx1, 0, tf.cast(w, tf.float32)-1)
      my0 = within_bounds(gy0, 0, tf.cast(h, tf.float32)-1)
      my1 = within_bounds(gy1, 0, tf.cast(h, tf.float32)-1)

      c00 = tf.expand_dims((gy1 - gy)*(gx1 - gx), axis=-1)
      c01 = tf.expand_dims((gy1 - gy)*(gx - gx0), axis=-1)
      c10 = tf.expand_dims((gy - gy0)*(gx1 - gx), axis=-1)
      c11 = tf.expand_dims((gy - gy0)*(gx - gx0), axis=-1)

      #clip for CPU (out_of_bound-error), optionnal on GPU (as corresponding m.. while be zeroed)
      gx0 = tf.clip_by_value(gx0, 0, tf.cast(w, tf.float32)-1)
      gx1 = tf.clip_by_value(gx1, 0, tf.cast(w, tf.float32)-1)
      gy0 = tf.clip_by_value(gy0, 0, tf.cast(h, tf.float32)-1)
      gy1 = tf.clip_by_value(gy1, 0, tf.cast(h, tf.float32)-1)

      g00 = tf.stack([gy0, gx0], axis=-1)
      g01 = tf.stack([gy0, gx1], axis=-1)
      g10 = tf.stack([gy1, gx0], axis=-1)
      g11 = tf.stack([gy1, gx1], axis=-1)

      m00 = tf.cast(tf.expand_dims(tf.logical_and(my0, mx0), axis=-1), tf.float32)
      m01 = tf.cast(tf.expand_dims(tf.logical_and(my0, mx1), axis=-1), tf.float32)
      m10 = tf.cast(tf.expand_dims(tf.logical_and(my1, mx0), axis=-1), tf.float32)
      m11 = tf.cast(tf.expand_dims(tf.logical_and(my1, mx1), axis=-1), tf.float32)

      x00 = tf.gather_nd(image, tf.cast(g00, dtype=tf.int32), batch_dims=1)
      x01 = tf.gather_nd(image, tf.cast(g01, dtype=tf.int32), batch_dims=1)
      x10 = tf.gather_nd(image, tf.cast(g10, dtype=tf.int32), batch_dims=1)
      x11 = tf.gather_nd(image, tf.cast(g11, dtype=tf.int32), batch_dims=1)

      output = c00 * x00 * m00 \
             + c01 * x01 * m01 \
             + c10 * x10 * m10 \
             + c11 * x11 * m11

      return output


def nearest_sampler(image, coords):
    with tf.name_scope("nearest_sampler"):
        _, h, w, _ = tf.unstack(tf.shape(image))

        gx, gy = tf.unstack(coords, axis=-1)

        # rescale x and y to [0, W-1/H-1]
        gx = (gx+1.0)/2.0  * tf.cast(w-1, tf.float32)
        gy = (gy+1.0)/2.0  * tf.cast(h-1, tf.float32)

        gx0 = tf.round(gx)
        gy0 = tf.round(gy)

        g00 = tf.stack([gy0, gx0], axis=-1)

        return tf.gather_nd(image, tf.cast(g00, dtype=tf.int32), batch_dims=1)



if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    import numpy as np

    for i in range(1000):

        test_size = 100

        grid_size = test_size
        feature_len = 1
        batch_size = test_size

        grid_sampling_size = test_size

        values = np.random.rand(batch_size, grid_size, grid_size, feature_len)

        t_values = np.transpose(values, (0, 3, 1, 2) )

        coords = np.random.rand(batch_size, grid_sampling_size, grid_sampling_size, 2) * 2 - 1
        coords = coords * 1.1

        values = values.astype(np.float32)
        coords = coords.astype(np.float32)
        t_values = t_values.astype(np.float32)

        tf_result = bilinear_sampler(values, coords)
        tf_result = tf_result.numpy()

        torch_result = F.grid_sample(torch.from_numpy(t_values), torch.from_numpy(coords),
            mode='bilinear', padding_mode='zeros', align_corners=True)


        torch_result = torch_result.view(batch_size, grid_sampling_size, grid_sampling_size, feature_len).numpy()

        diff = np.abs(tf_result - torch_result)

        print("diff", np.amax(diff), np.unravel_index(diff.argmax(), diff.shape))

        if np.amax(diff) > 1e-3:
            break
