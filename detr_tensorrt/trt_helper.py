import pycuda.driver as cuda
import tensorrt as trt

COCO_PANOPTIC_CLASS_NAMES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush', 'N/A', 'banner', 'blanket', 'N/A', 'bridge', 'N/A', 
    'N/A', 'N/A', 'N/A', 'cardboard', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 
    'counter', 'N/A', 'curtain', 'N/A', 'N/A', 'door-stuff', 'N/A', 'N/A', 'N/A', 'N/A', 
    'N/A', 'floor-wood', 'flower', 'N/A', 'N/A', 'fruit', 'N/A', 'N/A', 'gravel', 'N/A', 'N/A', 
    'house', 'N/A', 'light', 'N/A', 'N/A', 'mirror-stuff', 'N/A', 'N/A', 
    'N/A', 'N/A', 'net', 'N/A', 'N/A', 'pillow', 'N/A', 'N/A', 'platform', 
    'playingfield', 'N/A', 'railroad', 'river', 'road', 'N/A', 'roof', 'N/A', 'N/A', 
    'sand', 'sea', 'shelf', 'N/A', 'N/A', 'snow', 'N/A', 'stairs', 'N/A', 'N/A', 'N/A', 
    'N/A', 'tent', 'N/A', 'towel', 'N/A', 'N/A', 'wall-brick', 'N/A', 'N/A', 'N/A', 'wall-stone', 
    'wall-tile', 'wall-wood', 'water-other', 'N/A', 'window-blind', 'window-other', 'N/A', 'N/A', 
    'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 
    'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 
    'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 
    'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged', 'N/A', 
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 
    'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'backbround'
]

AGV_PANOPTIC_CLASS_NAMES = [
    "person", 'carpet', 'dirt', 'floor-mable', 'floor-other', 'floor-stone',
    'floor-tile', 'floor-wood', 'gravel', 'gournd-other', 'mud', 'pavement', 'platform', 'playingfield',
    'railroad', 'road', 'sand', 'snow', 'background'
]

def GiB(val):
    """Calculate Gibibit in bits, used to set workspace for TensorRT engine builder."""
    return val * 1 << 30

class HostDeviceMem(object):
    """
    Simple helper class to store useful data of an engine's binding

    Attributes:
    ----------
    host_mem: np.ndarray
        data stored in CPU
    device_mem: pycuda.driver.DeviceAllocation
        represent data pointer in GPU
    shape: tuple
    dtype: np dtype
    name: str
        name of the binding
    """
    def __init__(self, host_mem, device_mem, shape, dtype, name=""):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape
        self.dtype = dtype
        self.name = name

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(context, stream=None, is_async=True):
    """
    Read bindings' information in ExecutionContext, create pagelocked np.ndarray in CPU, 
    allocate corresponding memory in GPU.

    Returns:
    --------
    inputs: list[HostDeviceMem]
    outputs: list[HostDeviceMem]
    bindings: list[int]
        list of pointers in GPU for each bindings
    stream: pycuda.driver.Stream
        used for memory transfers between CPU-GPU
    """
    inputs = []
    outputs = []
    bindings = []
    if stream is None and is_async:
        stream = cuda.Stream()
    for binding in context.engine:
        binding_idx = context.engine.get_binding_index(binding)
        shape = context.get_binding_shape(binding_idx)
        size = trt.volume(shape) * context.engine.max_batch_size
        dtype = trt.nptype(context.engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if context.engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, shape, dtype, binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, shape, dtype, binding))
    return inputs, outputs, bindings, stream

def do_inference_async(context, bindings, inputs, outputs, stream):
    """
    Execute an TensorRT engine.

    Parameters:
    -----------
    context: tensorrt.IExecutionContext
    bindings: list[int]
        list of pointers in GPU for each bindings
    inputs: list[HostDeviceMem]
    outputs: list[HostDeviceMem]
    stream: pycuda.driver.Stream
        used for memory transfers between CPU-GPU

    Returns:
    --------
    list[np.ndarray] for each outputs of the engine
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    for out in outputs:
        out.host = out.host.reshape(out.shape)
    return [out.host for out in outputs]

def do_inference(context, bindings, inputs, outputs):
    """
    Execute an TensorRT engine.

    Parameters:
    -----------
    context: tensorrt.IExecutionContext
    bindings: list[int]
        list of pointers in GPU for each bindings
    inputs: list[HostDeviceMem]
    outputs: list[HostDeviceMem]
    stream: pycuda.driver.Stream
        used for memory transfers between CPU-GPU

    Returns:
    --------
    list[np.ndarray] for each outputs of the engine
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod(inp.device, inp.host) for inp in inputs]
    # Run inference.
    context.execute_v2(bindings=bindings)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh(out.host, out.device) for out in outputs]
    # # Synchronize the stream
    # stream.synchronize()
    # Return only the host outputs.
    for out in outputs:
        out.host = out.host.reshape(out.shape)
    return [out.host for out in outputs]