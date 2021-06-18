#include "ms_deform_im2col_kernel.h"
#include <cassert>
#include <stdio.h>
#include "cuda_fp16.h"
#include "NvInfer.h"


#define assertm(exp, msg) assert(((void)msg, exp))

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < n; i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;

int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}

template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_bilinear(const scalar_t* &bottom_data,
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const scalar_t &h, const scalar_t &w, const int &m, const int &c)
{
  const int h_low = floor(h);
  const int w_low = floor(w);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const scalar_t lh = h - h_low;
  const scalar_t lw = w - w_low;
  const scalar_t hh = 1 - lh, hw = 1 - lw;

  const int w_stride = nheads * channels;
  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ half ms_deform_attn_im2col_bilinear_half(const half* &bottom_data,
                                                   const int &height, const int &width, const int &nheads, const int &channels,
                                                   const half &h, const half &w, const int &m, const int &c)
{
  half one = __float2half(1.0f);
  half zero = __float2half(0.0f);

  const half h_low = hfloor(h);
  const half w_low = hfloor(w);
  const int h_high = hceil(h);
  const int w_high = hceil(w);

  
  const half lh = h - h_low;
  const half lw = w - w_low;
  const half hh = one - lh, hw = one - lw;

  const unsigned int w_stride = nheads * channels;
  const unsigned int h_stride = width * w_stride;
  const unsigned int h_low_ptr_offset = __half2uint_rd(h_low) * h_stride;
  const unsigned int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const unsigned int w_low_ptr_offset = __half2uint_rd(w_low) * w_stride;
  const unsigned int w_high_ptr_offset = w_low_ptr_offset + w_stride;
  const unsigned int base_ptr = m * channels + c;

  half v1 = __float2half(0.0f);
  if (h_low >= zero && w_low >= zero)
  {
    const int ptr1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  half v2 = 0;
  if (h_low >= zero && w_high <= width - 1)
  {
    const int ptr2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }
  half v3 = 0;
  if (h_high <= height - 1 && w_low >= zero)
  {
    const int ptr3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
    v3 = bottom_data[ptr3];
  }
  half v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
  {
    const int ptr4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;
    v4 = bottom_data[ptr4];
  }

  const half w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const half val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}


template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                const scalar_t *data_value,
                                                const int *data_spatial_shapes,
                                                const int *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_w = data_sampling_loc[data_loc_w_ptr];
        const scalar_t loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t h_im = loc_h * (spatial_h-1); //- 0.5;
        const scalar_t w_im = loc_w * (spatial_w-1); //- 0.5;

        if (h_im > -1 && w_im > -1 && h_im < spatial_h && w_im < spatial_w)
        {
          col += ms_deform_attn_im2col_bilinear<scalar_t>(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

__global__ void ms_deformable_im2col_gpu_kernel_half(const int n,
                                                const half *data_value,
                                                const int *data_spatial_shapes,
                                                const int *data_level_start_index,
                                                const half *data_sampling_loc,
                                                const half *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                half *data_col)
{
  half one(1.0f);
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    half *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_w_ptr = data_weight_ptr << 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    half col = 0;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_h_ptr = l_col << 1;
      const int spatial_h = data_spatial_shapes[spatial_h_ptr];
      const int spatial_w = data_spatial_shapes[spatial_h_ptr + 1];
      const half *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const half loc_w = data_sampling_loc[data_loc_w_ptr];
        const half loc_h = data_sampling_loc[data_loc_w_ptr + 1];
        const half weight = data_attn_weight[data_weight_ptr];

        const half h_im = loc_h * __int2half_rd(spatial_h-1); //- 0.5;
        const half w_im = loc_w * __int2half_rd(spatial_w-1); //- 0.5;

        if (h_im > -one && w_im > -one && h_im < __int2half_rd(spatial_h) && w_im < __int2half_rd(spatial_w))
        {
          col += ms_deform_attn_im2col_bilinear_half(data_value_ptr, spatial_h, spatial_w, num_heads, channels, h_im, w_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_w_ptr += 2;
      }
    }
    *data_col_ptr = col;
  }
}

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
)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;


  if(mDataType == DataType::kFLOAT)
  {
    // printf("Hey FLOAT \n");
    ms_deformable_im2col_gpu_kernel<float>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels, 
          static_cast<const float*>(data_value), 
          static_cast<const int*>(data_spatial_shapes), 
          static_cast<const int*>(data_level_start_index), 
          static_cast<const float*>(data_sampling_loc), 
          static_cast<const float*>(data_attn_weight),
          batch_size, 
          spatial_size,
          num_heads, 
          channels, 
          num_levels, 
          num_query, 
          num_point, 
          static_cast<float*>(data_col));
  }
  else if(mDataType == DataType::kHALF)
  {
    // printf("Hey HALF \n");
    ms_deformable_im2col_gpu_kernel_half
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads, 0, stream>>>(
          num_kernels, 
          static_cast<const half*>(data_value), 
          static_cast<const int*>(data_spatial_shapes), 
          static_cast<const int*>(data_level_start_index), 
          static_cast<const half*>(data_sampling_loc), 
          static_cast<const half*>(data_attn_weight),
          batch_size, 
          spatial_size,
          num_heads, 
          channels, 
          num_levels, 
          num_query, 
          num_point, 
          static_cast<half*>(data_col));

  }
  else return -1;

  return 0;
}