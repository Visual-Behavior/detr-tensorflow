import ctypes
import tensorrt as trt
import numpy as np
import os
import cv2
import argparse
import time
from numpy.lib.twodim_base import mask_indices

from detr_tensorrt.TRTExecutor import TRTExecutor, TRT_LOGGER
from detr_tensorrt.inference import normalized_images, get_model_inference

from detr_tf.data.coco import COCO_CLASS_NAME
from detr_tf.inference import numpy_bbox_to_image

BACKGROUND_CLASS = 91 # COCO background class

# Load custom plugin for deformable-detr
MS_DEFORM_IM2COL_PLUGIN_LIB = "./detr_tensorrt/plugins/ms_deform_im2col/build/libms_deform_im2col_trt.so"
ctypes.CDLL(MS_DEFORM_IM2COL_PLUGIN_LIB)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

def run_inference(model: TRTExecutor, normalized_image: np.ndarray):
    model.inputs[0].host = normalized_image
    model.execute()
    m_outputs = {out.name:out.host for out in model.outputs}
    p_bbox, p_labels, p_scores = get_model_inference(m_outputs, BACKGROUND_CLASS, threshold=0.2)
    return p_bbox, p_labels, p_scores

    
def main(engine_path):
    # Load TensorRT engine
    model = TRTExecutor(engine_path)
    model.print_bindings_info()

    # Read image
    input_shape = model.inputs[0].shape # (B, H, W, C)
    H, W = input_shape[1], input_shape[2]
    image = cv2.imread("images/test.jpeg")

    # Pre-process image
    model_input = cv2.resize(image, (W, H))
    model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
    model_input = normalized_images(model_input)

    # Run inference
    [model.execute() for i in range(3)] # GPU warm up

    tic = time.time()
    p_bbox, p_labels, p_scores = run_inference(model, model_input)
    toc = time.time()
    print(f"Inference latency: {(toc - tic)*1000} ms")

    image = image.astype(np.float32) / 255
    image = numpy_bbox_to_image(image, p_bbox, p_labels, p_scores, COCO_CLASS_NAME)

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_path", type=str, required=True)
    args = parser.parse_args()
    main(**vars(args))