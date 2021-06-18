#include <NvInferRuntimeCommon.h>
#include <vector>
#include <cassert>
#include <cstring>
#include <iostream>

#include "ms_deform_im2col_plugin.h"
#include "NvInfer.h"
#include "ms_deform_im2col_kernel.h"

using namespace std;

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace nvinfer1;

namespace {
    static const char* MS_DEFORM_IM2COL_PLUGIN_VERSION{"1"};
    static const char* MS_DEFORM_IM2COL_PLUGIN_NAME{"MsDeformIm2ColTRT"};
}

// Static class fields initialization
PluginFieldCollection MsDeformIm2ColCreator::mFC{};
std::vector<PluginField> MsDeformIm2ColCreator::mPluginAttributes;

// statically registers the Plugin Creator to the Plugin Registry of TensorRT
REGISTER_TENSORRT_PLUGIN(MsDeformIm2ColCreator);

// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}


MsDeformIm2Col::MsDeformIm2Col(const std::string name)
    : mLayerName(name)
{
}

MsDeformIm2Col::MsDeformIm2Col(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    im2col_step = readFromBuffer<int>(d);
    spatial_size = readFromBuffer<int>(d);
    num_heads = readFromBuffer<int>(d);
    channels = readFromBuffer<int>(d);
    num_levels = readFromBuffer<int>(d);
    num_query = readFromBuffer<int>(d);
    num_point = readFromBuffer<int>(d);
    mDataType = readFromBuffer<DataType>(d);

    assert(d == (a + length));
}

const char* MsDeformIm2Col::getPluginType() const noexcept
{
    return MS_DEFORM_IM2COL_PLUGIN_NAME;
}

const char* MsDeformIm2Col::getPluginVersion() const noexcept
{
    return MS_DEFORM_IM2COL_PLUGIN_VERSION;
}

int MsDeformIm2Col::getNbOutputs() const noexcept
{
    return 1;
}

Dims MsDeformIm2Col::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    int len_q = inputs[3].d[0];
    int num_heads = inputs[0].d[1];
    int head_dim = inputs[0].d[2];
    return Dims2(len_q, num_heads * head_dim);
}

int MsDeformIm2Col::initialize() noexcept
{
    return 0;
}

int MsDeformIm2Col::enqueue(int batchSize, const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int status = -1;
    // Launch CUDA kernel wrapper and save its return value
    status = ms_deform_im2col_inference(
        stream, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], 
        batchSize,spatial_size, num_heads, channels, 
        num_levels, num_query, num_point, outputs[0], mDataType);
    assert(status == 0);
    return status;
}

size_t MsDeformIm2Col::getSerializationSize() const noexcept
{
    // 7 int paramters
    return 7 * sizeof(int32_t) + sizeof(DataType);
}

void MsDeformIm2Col::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    writeToBuffer(d, im2col_step);
    writeToBuffer(d, spatial_size);
    writeToBuffer(d, num_heads);
    writeToBuffer(d, channels);
    writeToBuffer(d, num_levels);
    writeToBuffer(d, num_query);
    writeToBuffer(d, num_point);
    writeToBuffer(d, mDataType);

    assert(d == a + getSerializationSize());
}

void MsDeformIm2Col::terminate() noexcept {}

void MsDeformIm2Col::destroy() noexcept {
    // This gets called when the network containing plugin is destroyed
    delete this;
}

DataType MsDeformIm2Col::getOutputDataType(int32_t index, const DataType *inputTypes, int32_t nbInputs) const noexcept
{
    // only 1 output
    assert(index == 0);
    assert(nbInputs == 5);
    return inputTypes[0]; // return type of input tensor image
}

bool MsDeformIm2Col::isOutputBroadcastAcrossBatch(int32_t outputIndex, 
    const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept
{
    return false;
}

bool MsDeformIm2Col::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

void MsDeformIm2Col::configurePlugin(const PluginTensorDesc* in, int32_t nbInput, 
    const PluginTensorDesc* out, int32_t nbOutput) noexcept
{
    assertm(nbInput == 5, "Must provide 5 inputs: value, spatial_shape, start_index, sampling_locations, attn_weights\n");
    assertm(in[0].dims.nbDims == 3, "flatten_value must have shape (len_in, num_head, head_dim)\n");
    assertm(in[1].dims.nbDims == 2, "spatial_shapes must have shape (num_levels, 2)\n");
    assertm(in[2].dims.nbDims == 1, "start_index must have shape (num_levels, )\n");
    assertm(in[3].dims.nbDims == 5, "sampling_loc must have shape (len_q, num_head, num_levels, num_points, 2)\n");
    assertm(in[4].dims.nbDims == 4, "attn_weights must have shape (len_q, num_head, num_levels, num_points)\n");
    assertm(nbOutput == 1, "This layer has only one output.\n");
    
    im2col_step = 64;
    spatial_size = in[0].dims.d[0];
    num_heads = in[0].dims.d[1];
    channels = in[0].dims.d[2];
    num_levels = in[3].dims.d[2];
    num_query = in[3].dims.d[0];
    num_point = in[3].dims.d[3];
    mDataType = in[0].type;

    // cout << "DEBUG in[0].type: " <<  (int)in[0].type << endl;
    // cout << "MsDeformIm2Col DEBUG: im2col_step=" << im2col_step << endl;
    // cout << "MsDeformIm2Col DEBUG: spatial_size=" << spatial_size << endl;
    // cout << "MsDeformIm2Col DEBUG: num_heads=" << num_heads << endl;
    // cout << "MsDeformIm2Col DEBUG: channels=" << channels << endl;
    // cout << "MsDeformIm2Col DEBUG: num_levels=" << num_levels << endl;
    // cout << "MsDeformIm2Col DEBUG: num_query=" << num_query << endl;
    // cout << "MsDeformIm2Col DEBUG: num_point=" << num_point << endl;   
}

bool MsDeformIm2Col::supportsFormatCombination(int pos, const PluginTensorDesc* inOut, 
    int nbInputs, int nbOutputs) const noexcept
{
    bool ret;
    ret = inOut[pos].format == TensorFormat::kLINEAR;
    if((pos == 1) || (pos == 2))
    {
        return ret && (inOut[pos].type == DataType::kINT32);
    }
    else
    {   
        bool type_supported = (inOut[pos].type == DataType::kFLOAT) || (inOut[pos].type == DataType::kHALF);
        type_supported = type_supported && (inOut[pos].type == inOut[0].type);
        return ret && type_supported;
    }

}

IPluginV2Ext* MsDeformIm2Col::clone() const noexcept
{
    auto plugin = new MsDeformIm2Col(mLayerName);
    plugin->im2col_step = im2col_step;
    plugin->spatial_size = spatial_size;
    plugin->num_heads = num_heads;
    plugin->channels = channels;
    plugin->num_levels = num_levels;
    plugin->num_query = num_query;
    plugin->num_point = num_point;
    plugin->mDataType = mDataType;
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void MsDeformIm2Col::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MsDeformIm2Col::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

MsDeformIm2ColCreator::MsDeformIm2ColCreator()
{
    // Describe MsDeformIm2Col's required PluginField arguments

    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* MsDeformIm2ColCreator::getPluginName() const noexcept
{
    return MS_DEFORM_IM2COL_PLUGIN_NAME;
}

const char* MsDeformIm2ColCreator::getPluginVersion() const noexcept
{
    return MS_DEFORM_IM2COL_PLUGIN_VERSION;
}

const PluginFieldCollection* MsDeformIm2ColCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MsDeformIm2ColCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // const PluginField* fields = fc->fields;

    // Parse fields from PluginFieldCollection
    return new MsDeformIm2Col(name);
}

IPluginV2* MsDeformIm2ColCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MsDeformIm2Col::destroy()
    return new MsDeformIm2Col(name, serialData, serialLength);
}

void MsDeformIm2ColCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* MsDeformIm2ColCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
