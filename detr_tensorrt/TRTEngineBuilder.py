import tensorrt as trt
import os

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

class TRTEngineBuilder():
    """
    Work with TensorRT 8
    
    Helper class to build TensorRT engine from ONNX graph file (including weights). The graph must have defined input shape.
    For more detail, please see TensorRT Developer Guide:
    https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#python_topics
    """
    def __init__(self, onnx_file_path, FP16_allowed=False, INT8_allowed=False, strict_type=False, calibrator=None, logger=TRT_LOGGER):
        """
        Parameters:
        -----------
        onnx_file_path: str
            path to ONNX graph file
        FP16_allowed: bool
            Enable FP16 precision for engine builder
        INT8_allowed: bool
            Enable FP16 precision for engine builder, user must provide also a calibrator
        strict_type: bool
            Ensure that the builder understands to force the precision
        calibrator: extended instance from tensorrt.IInt8Calibrator
            Used for INT8 quantization
        """
        self.FP16_allowed = FP16_allowed
        self.INT8_allowed = INT8_allowed
        self.onnx_file_path = onnx_file_path
        self.calibrator = calibrator
        self.max_workspace_size = GiB(8)
        self.strict_type = strict_type
        self.logger = logger

    def set_workspace_size(self, workspace_size_GiB):
        self.max_workspace_size = GiB(workspace_size_GiB)

    def get_engine(self):
        """
        Setup engine builder, read ONNX graph and build TensorRT engine.
        """
        global network_creation_flag
        with trt.Builder(self.logger) as builder, builder.create_network(network_creation_flag) as network, trt.OnnxParser(network, self.logger) as parser:
            builder.max_batch_size = 1
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            # FP16
            if self.FP16_allowed:
                config.set_flag(trt.BuilderFlag.FP16)
            # INT8
            if self.INT8_allowed:
                raise NotImplementedError()
            if self.strict_type:
                config.set_flag(trt.BuilderFlag.STRICT_TYPES)

            # Load and build model 
            with open(self.onnx_file_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
                else:
                    print("ONNX file is loaded")
            print("Building cuda engine...")
            engine = builder.build_engine(network, config)
            if engine is None:
                Exception("TRT export engine error. Check log")
            print("Engine built")
        return engine

    def export_engine(self, engine_path):
        """Seriazlize TensorRT engine"""
        engine = self.get_engine()
        assert engine is not None, "Error while parsing engine from ONNX"
        with open(engine_path, "wb") as f:
                print("Serliaze and save as engine: " + engine_path)
                f.write(engine.serialize())
        print("Engine exported")


