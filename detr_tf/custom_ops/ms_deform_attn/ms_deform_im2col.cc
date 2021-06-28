#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"


#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

/*
:param values                        level, (N, H, W, num_heads, head_dim)
:param sampling_locations            level, (N, Len_q, num_heads, num_sampling_points, 2)
:param attention_weights              N, Len_q, num_heads, num_level, num_sampling_points
*/


REGISTER_OP("MsDeformIm2col")
    .Input("value: float")             // (N, Len_in, n_heads, d_model//n_heads)
    .Input("spatial_shapes: int32")    // (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
    .Input("level_start_index: int32") // (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
    .Input("sampling_loc: float")      // (N, Len_q, n_heads, n_levels, n_points, 2)
    .Input("attn_weight: float")       // (N, Len_q, num_heads,  n_level, num_sampling_points)
    .Attr("im2col_step:int = 64")
    .Output("col: float") // N, Len_q, num_heads*head_dim
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      auto batch_size = c->Dim(c->input(0), 0);
      auto num_heads = c->Dim(c->input(0), 2);
      auto channels = c->Dim(c->input(0), 3);
      auto num_query = c->Dim(c->input(3), 1);
      auto outChannels = c->MakeDim(round(c->Value(num_heads)*c->Value(channels)));
      c->set_output(0, c->MakeShape({batch_size, num_query, outChannels}));

      return Status::OK();
    });



REGISTER_OP("MsDeformIm2colGrad")
    .Input("value: float")             // (N, Len_in, n_heads, d_model//n_heads)
    .Input("spatial_shapes: int32")    // (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
    .Input("level_start_index: int32") // (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
    .Input("sampling_loc: float")      // (N, Len_q, n_heads, n_levels, n_points, 2)
    .Input("attn_weight: float")       // (N, Len_q, num_heads,  n_level, num_sampling_points)
    .Input("grad_output: float")       // N, Len_q, num_heads*head_dim
    .Attr("im2col_step:int = 64")
    .Output("grad_value: float")
    .Output("grad_sampling_loc: float")
    .Output("grad_attn_weight: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(3));
      c->set_output(2, c->input(4));
      return Status::OK();
    });


void ms_deformable_col2im_cuda(const GPUDevice& d,
                              const float* grad_col,
                              const float* value,
                              const int * spatial_shapes,
                              const int * level_start_index,
                              const float * sampling_loc,
                              const float * attn_weight,
                              const int batch_size,
                              const int spatial_size,
                              const int num_heads,
                              const int channels,
                              const int num_levels,
                              const int num_query,
                              const int num_point,
                              float* grad_value,
                              float* grad_sampling_loc,
                              float* grad_attn_weight);

void ms_deformable_im2col_cuda(const GPUDevice& d,
                              const float* value,
                              const int* spatial_shapes,
                              const int* level_start_index,
                              const float* sampling_loc,
                              const float* attn_weight,
                              const int batch_size,
                              const int spatial_size,
                              const int num_heads,
                              const int channels,
                              const int num_levels,
                              const int num_query,
                              const int num_point,
                              float* col);





template <typename scalar_t, typename Device>
class MsDeformIm2colOp : public OpKernel {
 public:
  explicit MsDeformIm2colOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("im2col_step", &im2col_step_));
    OP_REQUIRES(context, im2col_step_ >= 0,
                errors::InvalidArgument("Need im2col_step_ >= 0, got ",
                                        im2col_step_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& value                = context->input(0);
    const Tensor& spatial_shapes       = context->input(1);
    const Tensor& level_start_index    = context->input(2);
    const Tensor& sampling_loc         = context->input(3);
    const Tensor& attn_weight          = context->input(4);

    const int batch_size = value.dim_size(0);
    const int spatial_size = value.dim_size(1);
    const int num_heads = value.dim_size(2);
    const int channels = value.dim_size(3);
    const int num_levels = spatial_shapes.dim_size(0);
    const int num_query = sampling_loc.dim_size(1);
    const int num_point = sampling_loc.dim_size(4);

    const int im2col_step = std::min(batch_size, im2col_step_);

    Tensor* output_tensor = nullptr;

    TensorShape output_tensor_shape = TensorShape({batch_size, num_query, num_heads*channels});

    OP_REQUIRES_OK(context, context->allocate_output(0, output_tensor_shape, &output_tensor));
    auto col = output_tensor->flat<scalar_t>();


    // Call the cuda kernel launcher
    ms_deformable_im2col_cuda(context->eigen_gpu_device(),
                                value.flat<scalar_t>().data(),
                                spatial_shapes.flat<int>().data(),
                                level_start_index.flat<int>().data(),
                                sampling_loc.flat<scalar_t>().data(),
                                attn_weight.flat<scalar_t>().data(),
                                batch_size,
                                spatial_size,
                                num_heads,
                                channels,
                                num_levels,
                                num_query,
                                num_point,
                                col.data());
  }
 private:
  int im2col_step_;
};



template <typename scalar_t, typename Device>
class MsDeformIm2colGradOp : public OpKernel {
 public:
  explicit MsDeformIm2colGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("im2col_step", &im2col_step_));
    OP_REQUIRES(context, im2col_step_ >= 0,
                errors::InvalidArgument("Need im2col_step_ >= 0, got ",
                                        im2col_step_));
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& value                = context->input(0);
    const Tensor& spatial_shapes       = context->input(1);
    const Tensor& level_start_index    = context->input(2);
    const Tensor& sampling_loc         = context->input(3);
    const Tensor& attn_weight          = context->input(4);
    const Tensor& grad_output          = context->input(5);

    const int batch_size = value.dim_size(0);
    const int spatial_size = value.dim_size(1);
    const int num_heads = value.dim_size(2);
    const int channels = value.dim_size(3);
    const int num_levels = spatial_shapes.dim_size(0);
    const int num_query = sampling_loc.dim_size(1);
    const int num_point = sampling_loc.dim_size(4);

    Tensor* output_tensor_value        = nullptr;
    Tensor* output_tensor_sampling_loc = nullptr;
    Tensor* output_tensor_attn_weight  = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(0, value.shape(), &output_tensor_value));
    OP_REQUIRES_OK(context, context->allocate_output(1, sampling_loc.shape(), &output_tensor_sampling_loc));
    OP_REQUIRES_OK(context, context->allocate_output(2, attn_weight.shape(), &output_tensor_attn_weight));

    auto output_flat = output_tensor_value->flat<float>();


    // Call the cuda kernel launcher
    ms_deformable_col2im_cuda(context->eigen_gpu_device(),
                                  grad_output.flat<scalar_t>().data(),
                                  value.flat<scalar_t>().data(),
                                  spatial_shapes.flat<int>().data(),
                                  level_start_index.flat<int>().data(),
                                  sampling_loc.flat<scalar_t>().data(),
                                  attn_weight.flat<scalar_t>().data(),
                                  batch_size,
                                  spatial_size,
                                  num_heads,
                                  channels,
                                  num_levels,
                                  num_query,
                                  num_point,
                                  output_tensor_value->template flat<scalar_t>().data(),
                                  output_tensor_sampling_loc->template flat<scalar_t>().data(),
                                  output_tensor_attn_weight->template flat<scalar_t>().data());

  }
 private:
  int im2col_step_;
};






REGISTER_KERNEL_BUILDER(Name("MsDeformIm2col").Device(DEVICE_GPU), MsDeformIm2colOp<float, GPUDevice>);

REGISTER_KERNEL_BUILDER(Name("MsDeformIm2colGrad").Device(DEVICE_GPU), MsDeformIm2colGradOp<float, GPUDevice>);