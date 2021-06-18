# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(MsDeformIm2ColTRT LANGUAGES CXX CUDA)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wno-deprecated-declarations")

# Sets variable to a value if variable is unset.
macro(set_ifndef var val)
    if (NOT ${var})
        set(${var} ${val})
    endif()
    message(STATUS "Configurable variable ${var} set to ${${var}}")
endmacro()

# -------- CONFIGURATION --------
set_ifndef(TRT_LIB /home/ubuntu/TensorRT-8.0.0.3/lib)
set_ifndef(TRT_INCLUDE /home/ubuntu/TensorRT-8.0.0.3/include)
set_ifndef(CUDA_INC_DIR /usr/local/cuda/include)
set_ifndef(CUDA_ARCH_SM 70) # should be fine for Tesla V100

# Find dependencies:
message("\nThe following variables are derived from the values of the previous variables unless provided explicitly:\n")

# TensorRT's nvinfer lib
find_library(_NVINFER_LIB nvinfer HINTS ${TRT_LIB} PATH_SUFFIXES lib lib64)
set_ifndef(NVINFER_LIB ${_NVINFER_LIB})


# -------- BUILDING --------

# Add include directories
include_directories(${CUDA_INC_DIR} ${TRT_INCLUDE} ${CMAKE_SOURCE_DIR}/sources/)
message(STATUS "CUDA_INC_DIR: ${CUDA_INC_DIR}")
# Define plugin library target
add_library(ms_deform_im2col_trt MODULE
${CMAKE_SOURCE_DIR}/sources/ms_deform_im2col_kernel.cu
${CMAKE_SOURCE_DIR}/sources/ms_deform_im2col_kernel.h
${CMAKE_SOURCE_DIR}/sources/ms_deform_im2col_plugin.cpp
${CMAKE_SOURCE_DIR}/sources/ms_deform_im2col_plugin.h
)

# Use C++11
target_compile_features(ms_deform_im2col_trt PUBLIC cxx_std_11)

# Link TensorRT's nvinfer lib
target_link_libraries(ms_deform_im2col_trt PRIVATE ${NVINFER_LIB})

# We need to explicitly state that we need all CUDA files
# to be built with -dc as the member functions will be called by
# other libraries and executables (in our case, Python inference scripts)
set_target_properties(ms_deform_im2col_trt PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
)

# CUDA ARCHITECTURE
set_target_properties(ms_deform_im2col_trt PROPERTIES
CUDA_ARCHITECTURES "${CUDA_ARCH_SM}")
