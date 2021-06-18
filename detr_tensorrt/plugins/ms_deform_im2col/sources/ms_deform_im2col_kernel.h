#ifndef MS_DEFORM_IM2COL_KERNEL
#define MS_DEFORM_IM2COL_KERNEL

#include "NvInfer.h"
#include "cuda_fp16.h"

using namespace nvinfer1;


int ms_deform_im2col_inference(
    cudaStream_t stream,
    const void* data_value,
    const void* data_spatial_shapes,
    const void* data_level_start_index,
    const void* data_sampling_loc,
    const void* data_attn_weight,
    const int batch_size,
    const int spatial_size,
    const int num_heads,
    const int channels,
    const int num_levels,
    const int num_query,
    const int num_point,
    void* data_col,
    DataType mDataType
);

#endif

