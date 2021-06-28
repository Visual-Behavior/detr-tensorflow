
"""Cuda op Python library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import numpy as np
import tensorflow as tf

import math

from surroundnet.custom_ops.ms_deform_attn import ms_deform_im2col

from surroundnet.custom_ops.ms_deform_attn.ms_deform_attn import MSDeformAttnFunction



import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)

    #print(sampling_locations[:, :, :, 1, :, :])
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)

        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)

        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=True)
        # N_*M_, Lq_, P_, D_
        #tmp = sampling_value_l_.permute(0, 2, 3, 1)
        #print( f"sampling_value_l_{lid_}", tmp )
        # tmp = value_l_.permute(0, 2, 3, 1)
        # print("value", tmp)

        # tmp = sampling_value_l_.permute(0, 2, 3, 1)
        # print("sampled_values", tmp)

        # print("sampling_grid_l_", sampling_grid_l_)
        # exit()

        sampling_value_list.append(sampling_value_l_)
    #exit()
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, L_*P_)
    # (N_*M_, D_, Lq_, L_*P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
    #print(N_, M_, D_, Lq_, L_, P_)

    tmp = torch.stack(sampling_value_list, dim=-2).flatten(-2)

    # print("MSDeformAttnFunction_attention_weights", attention_weights.shape, attention_weights.view(N_, M_, 1, Lq_, L_, P_))
    # print("MSDeformAttnFunction_sampl", tmp.shape, tmp.view(N_, M_, D_, Lq_, L_, P_))

    #print("MSDeformAttnFunction_att*sampl", output.shape, output.view(N_, M_, D_, Lq_, L_, P_)[0, 2, :, 1, 3, 2])

    # (N_*M_, D_, Lq_) -> (N_, M_*D_, Lq_)
    output = output.sum(-1).view(N_, M_*D_, Lq_)
    # (N_, Lq_, M_*D_)
    return output.transpose(1, 2).contiguous()










N = 1

n_heads = 8
d_model = 256
size = np.array( (128, 128) )
Len_q = 13


# n_heads = 4
# d_model = 4
# size = np.array( (16, 16) )

# Len_q = 3

n_levels = 4
num_sampling_points = 4

values = list()
spatial_shapes = list()
level_start_index = [0]

for i in range(n_levels):
    value  = tf.random.uniform( shape=(N, int(size[0]), int(size[1]), n_heads, d_model//n_heads) ) * 0.01
    #value  = tf.ones( shape=(N, int(size[0]), int(size[1]), n_heads, d_model//n_heads) )
    values.append(value)
    spatial_shapes.append(size)
    level_start_index.append(size[0]*size[1] + level_start_index[-1])

    size = size//2

flatten_attn_weight  = tf.random.uniform( (N, Len_q, n_heads, n_levels, num_sampling_points) ) + 1e-5
flatten_attn_weight /= tf.reduce_sum(tf.reduce_sum(flatten_attn_weight, axis=-1, keepdims=True), axis=-2, keepdims=True)

#flatten_attn_weight  = tf.ones( (N, Len_q, n_heads, n_levels, num_sampling_points) )


flatten_sampling_loc = tf.random.uniform( (N, Len_q, n_heads, n_levels, num_sampling_points, 2), minval=-0.1, maxval=1.1, dtype=tf.float32 )
#flatten_sampling_loc = tf.ones( (N, Len_q, n_heads, n_levels, num_sampling_points, 2), dtype=tf.float32 ) *10 #(127+0.999) #* math.pi /10
#flatten_sampling_loc = tf.ones( (N, Len_q, n_heads, n_levels, num_sampling_points, 2), dtype=tf.float32 ) *0.5 #* math.pi /10
#flatten_sampling_loc = tf.ones( (N, Len_q, n_heads, n_levels, num_sampling_points, 2), dtype=tf.float32 ) * math.pi /10



level_start_index = np.array( level_start_index, dtype=np.int32 )

spatial_shapes = np.array( spatial_shapes, dtype=np.int32 )


with tf.GradientTape(persistent=True) as g:
    g.watch(flatten_sampling_loc)
    g.watch(values)
    g.watch(flatten_attn_weight)

    sampling_loc = tf.unstack(flatten_sampling_loc, axis=3) #(N, Len_q, n_heads, num_sampling_points)
    flatten_value = tf.concat( [tf.reshape(v, (N, -1, n_heads, d_model//n_heads) ) for v in values], axis=1)

    py_res = MSDeformAttnFunction(values, sampling_loc, flatten_attn_weight)

    res = ms_deform_im2col(
        flatten_value, # (N, Len_in, n_heads, d_model//n_heads)
        spatial_shapes,  # (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        level_start_index, # (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        flatten_sampling_loc, # (N, Len_q, n_heads, n_levels, n_points, 2)
        flatten_attn_weight # (N, Len_q, n_heads,  n_level, n_points)
    )

# Save tensors to npy for TensorRT plugin test
# np.save("./flatten_value.npy", flatten_value.numpy())
# np.save("./spatial_shapes.npy", spatial_shapes)
# np.save("./level_start_index.npy", level_start_index)
# np.save("./flatten_sampling_loc.npy", flatten_sampling_loc.numpy())
# np.save("./flatten_attn_weight.npy", flatten_attn_weight.numpy())
# np.save("./ms_deform_im2col_out.npy", res.numpy())

def check_value(name, py_grad, cu_grad):
    print(name, py_grad.shape)
    print("\t min value :", tf.reduce_min(py_grad), tf.reduce_min(cu_grad))
    print("\t max value :", tf.reduce_max(py_grad), tf.reduce_max(cu_grad))
    print("\t mean value :", tf.reduce_mean(py_grad), tf.reduce_mean(cu_grad))
    print("\t std value :", tf.math.reduce_std(py_grad), tf.math.reduce_std(cu_grad))
    abs_err = tf.reduce_max(tf.abs(py_grad - cu_grad))
    #coord = tf.math.argmax(tf.abs(py_grad - cu_grad))
    print("\t max abs error :", abs_err)
    abs_err = tf.reduce_mean(tf.abs(py_grad - cu_grad))
    print("\t mean abs error :", abs_err)
    #rel_err = tf.reduce_max(tf.abs(py_grad - cu_grad)/(tf.math.sqrt( tf.abs(py_grad)*tf.abs(cu_grad) ) + 1e-3) )
    #print("\t max rel error :", rel_err)

check_value("VALUE python / CUDA", py_res, res)
#print(py_res)
#print(res)

pytorch_res = ms_deform_attn_core_pytorch(
                    torch.from_numpy(flatten_value.numpy()),
                    spatial_shapes,
                    torch.from_numpy(flatten_sampling_loc.numpy()),
                    torch.from_numpy(flatten_attn_weight.numpy()) )

check_value("VALUE pytorch / tensorflow", pytorch_res, res)


#print(pytorch_res)
check_value("GRAD Sampling Loc", g.gradient(py_res, flatten_sampling_loc), g.gradient(res, flatten_sampling_loc))
check_value("GRAD Value", g.gradient(py_res, values[0]), g.gradient(res, values[0]))
check_value("GRAD Attention", g.gradient(py_res, flatten_attn_weight), g.gradient(res, flatten_attn_weight))


#(N, int(size[0]), int(size[1]), n_heads, d_model//n_heads)
#py_gvalue = g.gradient(py_res, values[0])
#cu_gvalue = g.gradient(res, values[0])
#print("cu_gvalue", cu_gvalue)
#print("cu_gvalue", cu_gvalue)

# args = [
#     flatten_value, # (N, Len_in, n_heads, d_model#n_heads)
#     spatial_shapes,  # (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
#     level_start_index, # (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
#     flatten_sampling_loc, # (N, Len_q, n_heads, n_levels, n_points, 2)
#     flatten_attn_weight # (N, Len_q, n_heads,  n_level, n_points)
# ]

#CUDA
#numerical, theoric =  tf.test.compute_gradient(ms_deform_im2col, args, delta=0.001)