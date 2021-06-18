import ctypes
import pycuda.autoinit as cuda_init
from surroundnet.detr.tensorrt.trt_helper import *
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
# trt.init_libnvinfer_plugins(None, "")

class TRTExecutor():
    """
    A helper class to execute a TensorRT engine.

    Attributes:
    -----------
    stream: pycuda.driver.Stream
    engine: tensorrt.ICudaEngine
    context: tensorrt.IExecutionContext
    inputs/outputs: list[HostDeviceMem]
        see trt_helper.py
    bindings: list[int] 
        pointers in GPU for each input/output of the engine
    dict_inputs/dict_outputs: dict[str, HostDeviceMem]
        key = input node name
        value = HostDeviceMem of corresponding binding

    """
    def __init__(self, engine_path=None, has_dynamic_shape=False, stream=None, engine=None):
        """
        Parameters:
        ----------
        engine_path: str
            path to serialized TensorRT engine
        has_dynamic_shape: bool
        stream: pycuda.driver.Stream
            if None, one will be created by allocate_buffers function
        """
        self.stream = stream
        if engine_path is not None:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                print("Reading engine  ...")
                self.engine = runtime.deserialize_cuda_engine(f.read())
                assert self.engine is not None, "Read engine failed"
                print("Engine loaded")
        elif engine is not None:
            self.engine = engine
        self.context = self.engine.create_execution_context()
        if not has_dynamic_shape:
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.context, self.stream)
            self.dict_inputs = {mem_obj.name:mem_obj for mem_obj in self.inputs}
            self.dict_outputs = {mem_obj.name:mem_obj for mem_obj in self.outputs}

    def print_bindings_info(self):
        print("ID / Name / isInput / shape / dtype")
        for i in range(self.engine.num_bindings):
            print(f"Binding: {i}, name: {self.engine.get_binding_name(i)}, input: {self.engine.binding_is_input(i)}, shape: {self.engine.get_binding_shape(i)}, dtype: {self.engine.get_binding_dtype(i)}")

    def execute(self):
        do_inference_async(
            self.context, 
            bindings=self.bindings, 
            inputs=self.inputs, 
            outputs=self.outputs, 
            stream=self.stream
        )

    def set_binding_shape(self, binding:int, shape:tuple):
        self.context.set_binding_shape(binding, shape)

    def allocate_mem(self):
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.context, self.stream)
        self.dict_inputs = {mem_obj.name:mem_obj for mem_obj in self.inputs}
        self.dict_outputs = {mem_obj.name:mem_obj for mem_obj in self.outputs}

class TRTExecutor_Sync():
    """
    A helper class to execute a TensorRT engine.

    Attributes:
    -----------
    engine: tensorrt.ICudaEngine
    context: tensorrt.IExecutionContext
    inputs/outputs: list[HostDeviceMem]
        see trt_helper.py
    bindings: list[int] 
        pointers in GPU for each input/output of the engine
    dict_inputs/dict_outputs: dict[str, HostDeviceMem]
        key = input node name
        value = HostDeviceMem of corresponding binding

    """
    def __init__(self, engine_path=None, has_dynamic_shape=False, engine=None):
        """
        Parameters:
        ----------
        engine_path: str
            path to serialized TensorRT engine
        has_dynamic_shape: bool
        """
        if engine_path is not None:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                print("Reading engine  ...")
                self.engine = runtime.deserialize_cuda_engine(f.read())
                assert self.engine is not None, "Read engine failed"
                print("Engine loaded")
        elif engine is not None:
            self.engine = engine
        self.context = self.engine.create_execution_context()
        if not has_dynamic_shape:
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.context, is_async=False)
            self.dict_inputs = {mem_obj.name:mem_obj for mem_obj in self.inputs}
            self.dict_outputs = {mem_obj.name:mem_obj for mem_obj in self.outputs}

    def print_bindings_info(self):
        print("ID / Name / isInput / shape / dtype")
        for i in range(self.engine.num_bindings):
            print(f"Binding: {i}, name: {self.engine.get_binding_name(i)}, input: {self.engine.binding_is_input(i)}, shape: {self.engine.get_binding_shape(i)}, dtype: {self.engine.get_binding_dtype(i)}")

    def execute(self):
        do_inference(
            self.context, 
            bindings=self.bindings, 
            inputs=self.inputs, 
            outputs=self.outputs,
        )

    def set_binding_shape(self, binding:int, shape:tuple):
        self.context.set_binding_shape(binding, shape)


    
    
