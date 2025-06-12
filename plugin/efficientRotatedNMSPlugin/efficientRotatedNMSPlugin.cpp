/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "efficientRotatedNMSPlugin.h"
#include "efficientRotatedNMSInference.h"

using namespace nvinfer1;
using nvinfer1::plugin::EfficientRotatedNMSPlugin;
using nvinfer1::plugin::EfficientRotatedNMSParameters;
using nvinfer1::plugin::EfficientRotatedNMSPluginCreator;

namespace
{
char const* const kEFFICIENT_ROTATED_NMS_PLUGIN_VERSION{"1"};
char const* const kEFFICIENT_ROTATED_NMS_PLUGIN_NAME{"EfficientRotatedNMS_TRT"};
} // namespace

/// @brief 静态注册插件创建器到注册表中
REGISTER_TENSORRT_PLUGIN(EfficientRotatedNMSPluginCreator);

EfficientRotatedNMSPlugin::EfficientRotatedNMSPlugin(EfficientRotatedNMSParameters param)
    : mParam(std::move(param))
{
}

EfficientRotatedNMSPlugin::EfficientRotatedNMSPlugin(void const* data, size_t length)
{
    deserialize(static_cast<int8_t const*>(data), length);
}

void EfficientRotatedNMSPlugin::deserialize(int8_t const* data, size_t length)
{
    auto const* d{data};
    mParam = read<EfficientRotatedNMSParameters>(d);
    PLUGIN_VALIDATE(d == data + length);
}

char const* EfficientRotatedNMSPlugin::getPluginType() const noexcept
{
    return kEFFICIENT_ROTATED_NMS_PLUGIN_NAME;
}

char const* EfficientRotatedNMSPlugin::getPluginVersion() const noexcept
{
    return kEFFICIENT_ROTATED_NMS_PLUGIN_VERSION;
}

int32_t EfficientRotatedNMSPlugin::getNbOutputs() const noexcept
{
    // Standard Plugin Implementation
    return 4;
}

int32_t EfficientRotatedNMSPlugin::initialize() noexcept
{
    if (!initialized)
    {
        /// 根据 GPU 计算能力动态调整 numSelectedBoxes 预选检测框数量
        int32_t device;
        /// 1. 查询设备属性
        CSC(cudaGetDevice(&device), STATUS_FAILURE);
        struct cudaDeviceProp properties;
        CSC(cudaGetDeviceProperties(&properties, device), STATUS_FAILURE);
        /// 2. 根据寄存器数量动态配置参数
        if (properties.regsPerBlock >= 65536)
        {
            // Most Devices 桌面/服务器级
            mParam.numSelectedBoxes = 2000;
        }
        else
        {
            // Jetson TX1/TX2  嵌入式设备
            mParam.numSelectedBoxes = 1000;
        }
        initialized = true;
    }
    return STATUS_SUCCESS;
}

void EfficientRotatedNMSPlugin::terminate() noexcept {}

size_t EfficientRotatedNMSPlugin::getSerializationSize() const noexcept
{
    return sizeof(EfficientRotatedNMSParameters);
}

void EfficientRotatedNMSPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, mParam);
    PLUGIN_ASSERT(d == a + getSerializationSize());
}

void EfficientRotatedNMSPlugin::destroy() noexcept
{
    delete this;
}

void EfficientRotatedNMSPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

char const* EfficientRotatedNMSPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType EfficientRotatedNMSPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    /// num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
    /// det_boxes = torch.randn(batch_size, max_output_boxes, 4)
    /// det_scores = torch.randn(batch_size, max_output_boxes)
    /// det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

    // On standard RotatedNMS, num_detections and detection_classes use integer outputs
    if (index == 0 || index == 3)
    {
        return nvinfer1::DataType::kINT32;
    }
    // All others should use the same datatype as the input
    /// 其他输出与输入的类型一致
    return inputTypes[0];
}

IPluginV2DynamicExt* EfficientRotatedNMSPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new EfficientRotatedNMSPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

DimsExprs EfficientRotatedNMSPlugin::getOutputDimensions(
    int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        DimsExprs out_dim;

        // When pad per class is set, the output size may need to be reduced:
        // i.e.: outputBoxes = min(outputBoxes, outputBoxesPerClass * numClasses)
        // As the number of classes may not be static, numOutputBoxes must be a dynamic
        // expression. The corresponding parameter can not be set at this time, so the
        // value will be calculated again in configurePlugin() and the param overwritten.
        /// 当启用 padOutputBoxesPerClass(按类别填充时) 时
        /// 最终输出框数 = min(预设最大框数, 每类最大框数 x 总类别数)
        IDimensionExpr const* numOutputBoxes = exprBuilder.constant(mParam.numOutputBoxes);
        if (mParam.padOutputBoxesPerClass && mParam.numOutputBoxesPerClass > 0)
        {
            IDimensionExpr const* numOutputBoxesPerClass = exprBuilder.constant(mParam.numOutputBoxesPerClass);
            IDimensionExpr const* numClasses = inputs[1].d[2]; ///< 从输入获取类别数
            numOutputBoxes = exprBuilder.operation(DimensionOperation::kMIN, *numOutputBoxes,
                *exprBuilder.operation(DimensionOperation::kPROD, *numOutputBoxesPerClass, *numClasses));
        }

        // Standard RotatedNMS
        PLUGIN_ASSERT(outputIndex >= 0 && outputIndex <= 3);

        // num_detections [batch_size, 1]
        if (outputIndex == 0)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0]; ///< batch_size
            out_dim.d[1] = exprBuilder.constant(1);
        }
        // detection_boxes [batch_size, numOutputBoxes, 5]
        else if (outputIndex == 1)
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = numOutputBoxes;
            out_dim.d[2] = exprBuilder.constant(5);
        }
        // detection_scores: outputIndex == 2  [batch_size, numOutputBoxes]
        // detection_classes: outputIndex == 3
        else if (outputIndex == 2 || outputIndex == 3)
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = numOutputBoxes;
        }

        return out_dim;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

bool EfficientRotatedNMSPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (inOut[pos].format != PluginFormat::kLINEAR)
    {
        return false;
    }

    PLUGIN_ASSERT(nbInputs == 2 || nbInputs == 3);
    PLUGIN_ASSERT(nbOutputs == 4);
    if (nbInputs == 2)
    {
        PLUGIN_ASSERT(0 <= pos && pos <= 5);
    }
    if (nbInputs == 3)
    {
        PLUGIN_ASSERT(0 <= pos && pos <= 6);
    }

    // num_detections and detection_classes output: int32_t
    int32_t const posOut = pos - nbInputs;
    if (posOut == 0 || posOut == 3)
    {
        return inOut[pos].type == DataType::kINT32 && inOut[pos].format == PluginFormat::kLINEAR;
    }

    // all other inputs/outputs: fp32 or fp16
    return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT)
        && (inOut[0].type == inOut[pos].type);
}

void EfficientRotatedNMSPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        /// 1. 验证输入输出个数
        // Accepts two or three inputs
        // If two inputs: [0] boxes, [1] scores
        // If three inputs: [0] boxes, [1] scores, [2] anchors
        PLUGIN_ASSERT(nbInputs == 2 || nbInputs == 3);
        PLUGIN_ASSERT(nbOutputs == 4);

        mParam.datatype = in[0].desc.type;

        // Shape of scores input should be
        // [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        PLUGIN_ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));
        mParam.numScoreElements = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        mParam.numClasses = in[1].desc.dims.d[2];

        // When pad per class is set, the total output boxes size may need to be reduced.
        // This operation is also done in getOutputDimension(), but for dynamic shapes, the
        // numOutputBoxes param can't be set until the number of classes is fully known here.
        if (mParam.padOutputBoxesPerClass && mParam.numOutputBoxesPerClass > 0)
        {
            if (mParam.numOutputBoxesPerClass * mParam.numClasses < mParam.numOutputBoxes)
            {
                mParam.numOutputBoxes = mParam.numOutputBoxesPerClass * mParam.numClasses;
            }
        }

        // Shape of boxes input should be
        // [batch_size, num_boxes, 5] or [batch_size, num_boxes, 1, 5] or [batch_size, num_boxes, num_classes, 5]
        PLUGIN_ASSERT(in[0].desc.dims.nbDims == 3 || in[0].desc.dims.nbDims == 4);
        if (in[0].desc.dims.nbDims == 3)
        {
            PLUGIN_ASSERT(in[0].desc.dims.d[2] == 5);
            mParam.shareLocation = true;
            mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2];
        }
        else
        {
            mParam.shareLocation = (in[0].desc.dims.d[2] == 1);
            PLUGIN_ASSERT(in[0].desc.dims.d[2] == mParam.numClasses || mParam.shareLocation);
            PLUGIN_ASSERT(in[0].desc.dims.d[3] == 5);
            mParam.numBoxElements = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        }
        mParam.numAnchors = in[0].desc.dims.d[1];

        if (nbInputs == 2)
        {
            // Only two inputs are used, disable the fused box decoder
            mParam.boxDecoder = false;
        }
        if (nbInputs == 3)
        {
            // All three inputs are used, enable the box decoder
            // Shape of anchors input should be
            // Constant shape: [1, numAnchors, 4] or [batch_size, numAnchors, 4]
            PLUGIN_ASSERT(in[2].desc.dims.nbDims == 3);
            mParam.boxDecoder = true;
            mParam.shareAnchors = (in[2].desc.dims.d[0] == 1);
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
}

size_t EfficientRotatedNMSPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    /// scores [batch_size, num_boxes, num_class]
    int32_t batchSize = inputs[1].dims.d[0];
    int32_t numScoreElements = inputs[1].dims.d[1] * inputs[1].dims.d[2];
    int32_t numClasses = inputs[1].dims.d[2];
    return EfficientRotatedNMSWorkspaceSize(batchSize, numScoreElements, numClasses, mParam.datatype);
}

int32_t EfficientRotatedNMSPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /* outputDesc */,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        /// 1. 检查指针有效性
        PLUGIN_VALIDATE(inputDesc != nullptr && inputs != nullptr && outputs != nullptr && workspace != nullptr);

        mParam.batchSize = inputDesc[0].dims.d[0];

        // Standard RotatedNMS Operation
        /// 2. 获取输入指针
        void const* const boxesInput = inputs[0];
        void const* const scoresInput = inputs[1];
        void const* const anchorsInput = mParam.boxDecoder ? inputs[2] : nullptr;

        /// 3. 获取输出指针
        void* numDetectionsOutput = outputs[0];
        void* nmsBoxesOutput = outputs[1];
        void* nmsScoresOutput = outputs[2];
        void* nmsClassesOutput = outputs[3];

        /// 4. 推理
        return EfficientRotatedNMSInference(mParam, boxesInput, scoresInput, anchorsInput, numDetectionsOutput, nmsBoxesOutput,
            nmsScoresOutput, nmsClassesOutput, workspace, stream);
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return -1;
}

// Standard RotatedNMS Plugin Operation

EfficientRotatedNMSPluginCreator::EfficientRotatedNMSPluginCreator()
    : mParam{}
{
    mPluginAttributes.clear();
    /// 定义插件支持的参数列表
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iou_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_output_boxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("background_class", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_activation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("class_agnostic", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("box_coding", nullptr, PluginFieldType::kINT32, 1));
    /// 将参数列表包装为 TensorRT 标准格式
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* EfficientRotatedNMSPluginCreator::getPluginName() const noexcept
{
    return kEFFICIENT_ROTATED_NMS_PLUGIN_NAME;
}

char const* EfficientRotatedNMSPluginCreator::getPluginVersion() const noexcept
{
    return kEFFICIENT_ROTATED_NMS_PLUGIN_VERSION;
}

PluginFieldCollection const* EfficientRotatedNMSPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2DynamicExt* EfficientRotatedNMSPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    try
    {
        /// 1. 参数有效性验证
        PLUGIN_VALIDATE(fc != nullptr);
        PluginField const* fields = fc->fields;
        PLUGIN_VALIDATE(fields != nullptr);

        /// 2. 必要参数存在性检查
        plugin::validateRequiredAttributesExist({"score_threshold", "iou_threshold", "max_output_boxes",
                                                    "background_class", "score_activation", "box_coding"},
            fc);

        /// 3. 参数解析与类型验证
        for (int32_t i{0}; i < fc->nbFields; ++i)
        {
            char const* attrName = fields[i].name;
            if (!strcmp(attrName, "score_threshold"))
            {
                /// 4. 验证类型+范围
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                auto const scoreThreshold = *(static_cast<float const*>(fields[i].data));
                PLUGIN_VALIDATE(scoreThreshold >= 0.0F);
                /// 5. 填充 mParam
                mParam.scoreThreshold = scoreThreshold;
            }
            if (!strcmp(attrName, "iou_threshold"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kFLOAT32);
                auto const iouThreshold = *(static_cast<float const*>(fields[i].data));
                PLUGIN_VALIDATE(iouThreshold > 0.0F);
                mParam.iouThreshold = iouThreshold;
            }
            if (!strcmp(attrName, "max_output_boxes"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                auto const numOutputBoxes = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(numOutputBoxes > 0);
                mParam.numOutputBoxes = numOutputBoxes;
            }
            if (!strcmp(attrName, "background_class"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                mParam.backgroundClass = *(static_cast<int32_t const*>(fields[i].data));
            }
            if (!strcmp(attrName, "score_activation"))
            {
                auto const scoreSigmoid = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(scoreSigmoid == 0 || scoreSigmoid == 1);
                mParam.scoreSigmoid = static_cast<bool>(scoreSigmoid);
            }
            if (!strcmp(attrName, "class_agnostic"))
            {
                auto const classAgnostic = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(classAgnostic == 0 || classAgnostic == 1);
                mParam.classAgnostic = static_cast<bool>(classAgnostic);
            }
            if (!strcmp(attrName, "box_coding"))
            {
                PLUGIN_VALIDATE(fields[i].type == PluginFieldType::kINT32);
                auto const boxCoding = *(static_cast<int32_t const*>(fields[i].data));
                PLUGIN_VALIDATE(boxCoding == 0 || boxCoding == 1);
                mParam.boxCoding = boxCoding;
            }
        }

        /// 6. 创建插件实例
        auto* plugin = new EfficientRotatedNMSPlugin(mParam);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* EfficientRotatedNMSPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call EfficientNMSPlugin::destroy()
        /// 1. 动态内存分配
        auto* plugin = new EfficientRotatedNMSPlugin(serialData, serialLength);
        /// 2. 设置命名空间
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
