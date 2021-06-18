#ifndef MS_DEFORM_IM2COL_TRT_PLUGIN_H
#define MS_DEFORM_IM2COL_TRT_PLUGIN_H

#include "NvInferPlugin.h"
#include <string>
#include <vector>


using namespace nvinfer1;

// One of the preferred ways of making TensorRT to be able to see
// our custom layer requires extending IPluginV2 and IPluginCreator classes.
// For requirements for overriden functions, check TensorRT API docs.

class MsDeformIm2Col : public IPluginV2IOExt
{
public:
    MsDeformIm2Col(const std::string name);

    MsDeformIm2Col(const std::string name, const void* data, size_t length);

    // It doesn't make sense to make MsDeformIm2Col without arguments, so we delete default constructor.
    MsDeformIm2Col() = delete;

    int getNbOutputs() const noexcept override;

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;

    int initialize() noexcept override;

    void terminate() noexcept override;

    size_t getWorkspaceSize(int) const noexcept override { return 0; };

    int enqueue(int batchSize, const void* const* inputs, void* const* outputs, void* workspace, 
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    const char* getPluginType() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    void destroy() noexcept override;

    IPluginV2Ext* clone() const noexcept override;

    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(int32_t index, const nvinfer1::DataType *inputTypes, int32_t nbInputs) const noexcept override;

    bool isOutputBroadcastAcrossBatch(int32_t outputIndex, const bool* inputIsBroadcasted, int32_t nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

    void configurePlugin(const PluginTensorDesc* in, int32_t nbInput, const PluginTensorDesc* out, int32_t nbOutput) noexcept override;

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const noexcept override;

private:
    const std::string mLayerName;
    std::string mNamespace;
    int im2col_step;
    int spatial_size; 
    int num_heads; 
    int channels; 
    int num_levels; 
    int num_query; 
    int num_point; 
    DataType mDataType;
    // DataType mDataType = DataType::kHALF;
};

class MsDeformIm2ColCreator : public IPluginCreator
{
public:
    MsDeformIm2ColCreator();

    const char* getPluginName() const noexcept override;

    const char* getPluginVersion() const noexcept override;

    const PluginFieldCollection* getFieldNames() noexcept override;

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;

    IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    
    void setPluginNamespace(const char* pluginNamespace) noexcept override;

    const char* getPluginNamespace() const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};

#endif
