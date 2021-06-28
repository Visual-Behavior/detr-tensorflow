import os.path
import tensorflow as tf

if tf.test.is_built_with_cuda():
    _cuda_op_module = tf.load_op_library(os.path.join(
        tf.compat.v1.resource_loader.get_data_files_path(), 'ms_deform_im2col.so'))
    ms_deform_im2col = _cuda_op_module.ms_deform_im2col


    @tf.RegisterGradient("MsDeformIm2col")
    def _zero_out_grad(op, grad):
        grad_value, grad_sampling_loc, grad_attn_weight =  _cuda_op_module.ms_deform_im2col_grad(
            op.inputs[0],
            op.inputs[1],
            op.inputs[2],
            op.inputs[3],
            op.inputs[4],
            grad
        )

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight


else:
    raise ValueError("Trying to load cuda ms_deform_im2col without cuda support")