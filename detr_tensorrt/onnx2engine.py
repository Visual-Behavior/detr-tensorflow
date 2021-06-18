import argparse
import ctypes
import tensorrt as trt
import os
import onnx
import numpy as np
import onnx_graphsurgeon as gs

from TRTEngineBuilder import TRTEngineBuilder, TRT_LOGGER
from common import GiB

MS_DEFORM_IM2COL_PLUGIN_LIB = "./detr_tensorrt/plugins/ms_deform_im2col/build/libms_deform_im2col_trt.so"
ctypes.CDLL(MS_DEFORM_IM2COL_PLUGIN_LIB)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list


def print_graph_io(graph):
    # Print inputs:
    print(" ===== Inputs =====")
    for i in graph.inputs:
        print(i)
    # Print outputs:
    print(" ===== Outputs =====")
    for i in graph.outputs:
        print(i)


def io_name_handler(graph: gs.Graph):
    len_suffix = len("tf_op_layer_")
    for out in graph.outputs:
        out.name = out.name[len_suffix:]


def get_node_by_name(name, onnx_graph: gs.Graph):
    for n in onnx_graph.nodes:
        if name in n.name:
            return n
    return None


def get_nodes_by_op(op_name, onnx_graph):
    nodes = []
    for n in onnx_graph.nodes:
        if n.op == op_name:
            nodes.append(n)
    return nodes


def fix_graph_detr(graph: gs.Graph):
    # === Fix Pad 2 in Resnet backbone ===
    # TensorRT supports padding only on 2 innermost dimensions
    resnet_pad2 = get_node_by_name(
        "detr/detr_finetuning/detr/backbone/pad2/Pad", graph)
    resnet_pad2.inputs[1] = gs.Constant(
        "pad2/pads_input", np.array([0, 0, 1, 1, 0, 0, 1, 1]))
    graph.cleanup()
    graph.toposort()
    return graph


def fix_graph_deformable_detr(graph: gs.Graph):
    # === Fix Pad 2 in Resnet backbone ===
    # TensorRT supports padding only on 2 innermost dimensions
    resnet_pad2 = get_node_by_name(
        "deformable-detr/deformable_detr/detr_core/backbone/pad2/Pad", graph)
    resnet_pad2.inputs[1] = gs.Constant(
        "pad2/pads_input", np.array([0, 0, 1, 1, 0, 0, 1, 1]))

    # ======= Add nodes for MsDeformIm2ColTRT ===========
    tf_im2col_nodes = get_nodes_by_op("MsDeformIm2col", graph)

    spatial_shape_np = tf_im2col_nodes[0].inputs[1].values.reshape((1, -1, 2))
    spatial_shape_tensor = gs.Constant(
        name="MsDeformIm2Col_spatial_shape",
        values=spatial_shape_np)

    start_index_np = tf_im2col_nodes[0].inputs[2].values.reshape((1, -1))
    start_index_tensor = gs.Constant(
        name="MsDeformIm2Col_start_index",
        values=start_index_np)

    def handle_ops_MsDeformIm2ColTRT(graph: gs.Graph, node: gs.Node):
        inputs = node.inputs
        inputs.pop(1)
        inputs.pop(1)
        inputs.insert(1, start_index_tensor)
        inputs.insert(1, spatial_shape_tensor)
        outputs = node.outputs
        graph.layer(
            op="MsDeformIm2ColTRT",
            name=node.name + "_trt",
            inputs=inputs,
            outputs=outputs)

    for n in tf_im2col_nodes:
        handle_ops_MsDeformIm2ColTRT(graph, n)
        # Detach old node from graph
        n.inputs.clear()
        n.outputs.clear()
        graph.nodes.remove(n)

    return graph


def fix_onnx_graph(graph: gs.Graph, model: str):
    if model == "detr":
        return fix_graph_detr(graph)
    elif model == "deformable-detr":
        return fix_graph_deformable_detr(graph)


def main(onnx_dir: str, model: str, precision: str, verbose: bool, **args):
    print(model)
    onnx_path = os.path.join(onnx_dir, model + ".onnx")
    print(onnx_path)

    graph = gs.import_onnx(onnx.load(onnx_path))
    graph.toposort()

    # === Change graph IO names
    # print_graph_io(graph)
    io_name_handler(graph)
    print_graph_io(graph)

    # === Fix graph to adapt to TensorRT
    graph = fix_onnx_graph(graph, model)

    # === Export adapted onnx for TRT engine
    adapted_onnx_path = os.path.join(onnx_dir, model + "_trt.onnx")
    onnx.save(gs.export_onnx(graph), adapted_onnx_path)

    # === Build engine
    if verbose:
        trt_logger = trt.Logger(trt.Logger.VERBOSE)
    else:
        trt_logger = trt.Logger(trt.Logger.WARNING)

    builder = TRTEngineBuilder(adapted_onnx_path, logger=trt_logger)

    if precision == "FP32":
        pass
    if precision == "FP16":
        builder.FP16_allowed = True
        builder.strict_type = True
    if precision == "MIX":
        builder.FP16_allowed = True
        builder.strict_type = False

    builder.export_engine(os.path.join(
        onnx_dir, model + f"_{precision.lower()}.engine"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default="detr",
                        help="detr/deformable-detr")
    parser.add_argument("--precision", type=str,
                        default="FP16", help="FP32/FP16/MIX")
    parser.add_argument('--onnx_dir', type=str, default=None,
                        help="path to dir containing the \{model_name\}.onnx file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print out TensorRT log of all levels")
    args = parser.parse_args()

    if args.onnx_dir is None:
        args.onnx_dir = os.path.join(
            "weights", args.model, args.model + "_trt")
    # for plugin in PLUGIN_CREATORS:
    #     print(plugin.name, plugin.plugin_version)
    main(**vars(args))
