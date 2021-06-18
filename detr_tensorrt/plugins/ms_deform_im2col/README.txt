To build the plugin:
mkdir build && cd build
cmake .. && make -j

NOTE: If any of the dependencies are not installed in their default locations, you can manually specify them. For example:

cmake .. -DPYBIND11_DIR=/path/to/pybind11/
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda-x.x/bin/nvcc  (Or adding /path/to/nvcc into $PATH)
         -DCUDA_INC_DIR=/usr/local/cuda-x.x/include/  (Or adding /path/to/cuda/include into $CPLUS_INCLUDE_PATH)
         -DPYTHON3_INC_DIR=/usr/include/python3.6/
         -DTRT_LIB=/path/to/tensorrt/lib/
         -DTRT_INCLUDE=/path/to/tensorrt/include/
         -DCUDA_ARCH_SM=70

Check matching sm for Nvidia GPU:
https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/