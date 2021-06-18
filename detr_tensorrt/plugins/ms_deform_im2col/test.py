import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import ctypes
import re
import time

from surroundnet.detr.tensorrt.TRTExecutor import TRTExecutor

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

current_path = os.path.dirname(os.path.abspath(__file__))

MS_DEFORM_IM2COL_PLUGIN_LIB = "./detr_tensorrt/plugins/ms_deform_im2col/build/libms_deform_im2col_trt.so"
ctypes.CDLL(MS_DEFORM_IM2COL_PLUGIN_LIB)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def GiB(val):
    return val * 1 << 30

def camel_to_snake(name):
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            plugin = plugin_creator.create_plugin(camel_to_snake(plugin_name), None)
    if plugin is None:
        raise Exception(f"plugin {plugin_name} not found")
    return plugin

def build_test_engine(input_shape, dtype=trt.float32):
    num_level = input_shape["flatten_sampling_loc"][3]

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network:
        builder.max_batch_size = 1
        config = builder.create_builder_config()
        config.max_workspace_size = GiB(5)
        if dtype == trt.float16:
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        input_flatten_value = network.add_input(
            name="input_flatten_value", dtype=dtype, shape=input_shape["flatten_value"])
        input_spatial_shapes = network.add_input(
            name="input_spatial_shapes", dtype=trt.int32, shape=(1, num_level, 2))
        input_start_index = network.add_input(
            name="input_start_index", dtype=trt.int32, shape=(1, num_level))
        input_flatten_sampling_loc = network.add_input(
            name="input_flatten_sampling_loc", dtype=dtype, shape=input_shape["flatten_sampling_loc"])
        input_flatten_attn_weight = network.add_input(
            name="input_flatten_attn_weight", dtype=dtype, shape=input_shape["flatten_attn_weight"])
        
        ms_deform_im2col_node = network.add_plugin_v2(
            inputs=[
                input_flatten_value, input_spatial_shapes,
                input_start_index, input_flatten_sampling_loc,
                input_flatten_attn_weight],
            plugin=get_trt_plugin("MsDeformIm2ColTRT")
        )
        ms_deform_im2col_node.name = "ms_deform_im2col_node"
        ms_deform_im2col_node.get_output(0).name = "im2col_output"

        network.mark_output(ms_deform_im2col_node.get_output(0))

        return builder.build_engine(network, config)

def get_target_test_tensors(dtype=np.float32):
    test_dir = os.path.join(current_path, "test_tensors")
    test_tensors = {}
    test_shapes = {}
    for filename in os.listdir(test_dir):
        tensor_name = filename.split(".")[0]
        test_tensors[tensor_name] = np.load(os.path.join(test_dir, filename))

        if test_tensors[tensor_name].dtype == np.int64:
            test_tensors[tensor_name] = test_tensors[tensor_name].astype(np.int32)
        elif test_tensors[tensor_name].dtype == np.float32:
            test_tensors[tensor_name] = test_tensors[tensor_name].astype(dtype)

        test_shapes[tensor_name] = test_tensors[tensor_name].shape
    return test_tensors, test_shapes


if __name__ == "__main__":
    # for plugin in PLUGIN_CREATORS:
    #     print(plugin.name, plugin.plugin_version)

    test_tensors, test_shapes = get_target_test_tensors()
    for key in test_tensors:
        print(key, test_tensors[key].shape, test_tensors[key].dtype)
    test_engine = build_test_engine(test_shapes, dtype=trt.float16)

    trt_model = TRTExecutor(engine=test_engine)
    trt_model.print_bindings_info()

    trt_model.inputs[0].host = test_tensors["flatten_value"].astype(np.float16)
    trt_model.inputs[1].host = test_tensors["spatial_shapes"]
    trt_model.inputs[2].host = test_tensors["level_start_index"][:4].copy()
    trt_model.inputs[3].host = test_tensors["flatten_sampling_loc"].astype(np.float16)
    trt_model.inputs[4].host = test_tensors["flatten_attn_weight"].astype(np.float16)


    trt_model.execute()

    N = 1000
    tic = time.time()
    [trt_model.execute() for i in range(N)]
    toc = time.time()

    diff = test_tensors["output"] - trt_model.outputs[0].host
    print(np.abs(diff).mean())
    print(f"Execution time: {(toc - tic)/N*1000} ms")
    