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
#ifndef TRT_EFFICIENT_ROTATED_NMS_PLUGIN_H
#define TRT_EFFICIENT_ROTATED_NMS_PLUGIN_H

#include <vector>
#include <string>

#include <NvInferPlugin.h>

#include "common/plugin.h"
#include "efficientRotatedNMSParameters.h"

namespace nvinfer1
{
namespace plugin
{

class EfficientRotatedNMSPlugin : public IPluginV2DynamicExt
{
public:
    /// @brief 直接使用参数创建插件实例，createPlugin() 中调用
    /// @param param 
    explicit EfficientRotatedNMSPlugin(EfficientRotatedNMSParameters param);

    /// @brief 从序列化数据中创建插件实例
    /// @param data 
    /// @param length 
    EfficientRotatedNMSPlugin(void const* data, size_t length);

    ~EfficientRotatedNMSPlugin() override = default;

    // IPluginV2 methods

    /// @brief 返回插件类型
    /// @return "EfficientRotatedNMS"
    char const* getPluginType() const noexcept override;

    /// @brief 返回插件版本
    /// @return "1"
    char const* getPluginVersion() const noexcept override;

    /// @brief 返回输出张量数量
    /// @return 
    int32_t getNbOutputs() const noexcept override;

    /// @brief 初始化插件资源
    /// @return 
    int32_t initialize() noexcept override;

    /// @brief 销毁插件资源
    void terminate() noexcept override;

    /// @brief 获取序列化参数大小
    /// @return 
    size_t getSerializationSize() const noexcept override;

    /// @brief 序列化插件参数
    /// @param buffer 
    void serialize(void* buffer) const noexcept override;

    /// @brief 删除插件实例 delete this
    void destroy() noexcept override;

    /// @brief 设置插件命名空间
    /// @param libNamespace 
    void setPluginNamespace(char const* libNamespace) noexcept override;

    /// @brief 获取插件命名空间
    /// @return 
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    
    /// @brief 指定输出张量的数据类型
    /// @param index 索引
    /// @param inputType 输入数据类型
    /// @param nbInputs 输入数据个数
    /// @return 
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods

    /// @brief 深拷贝插件，用于多线程
    /// @return 
    IPluginV2DynamicExt* clone() const noexcept override;

    /// @brief 根据输入动态计算输出维度
    /// @param outputIndex 
    /// @param inputs 
    /// @param nbInputs 
    /// @param exprBuilder 
    /// @return 
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    
    /// @brief 检查输入/输出的数据格式支持
    /// @param pos 
    /// @param inOut 
    /// @param nbInputs 
    /// @param nbOutputs 
    /// @return 
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    
    /// @brief 验证输入/输出的形状和数据类型
    /// @param in bbox score
    /// @param nbInputs 2
    /// @param out det_classes det_scores det_boxes num_dets
    /// @param nbOutputs 4
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    /// @brief 返回插件所需的临时 GPU 工作空间大小
    /// @param inputs 
    /// @param nbInputs 
    /// @param outputs 
    /// @param nbOutputs 
    /// @return 
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    
    /// @brief 执行
    /// @param inputDesc 
    /// @param outputDesc 
    /// @param inputs 输入数据
    /// @param outputs 输出数据
    /// @param workspace 
    /// @param stream CUDA 流
    /// @return 
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

protected:
    EfficientRotatedNMSParameters mParam{}; ///< NMS 算法参数
    bool initialized{false};                ///< 初始化标志
    std::string mNamespace;                 ///< 插件命名空间

private:
    void deserialize(int8_t const* data, size_t length);
};

// Standard RotatedNMS Plugin Operation
class EfficientRotatedNMSPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    EfficientRotatedNMSPluginCreator();
    ~EfficientRotatedNMSPluginCreator() override = default;

    /// @brief 返回插件名称
    /// @return 
    char const* getPluginName() const noexcept override;

    /// @brief 返回插件版本
    /// @return 
    char const* getPluginVersion() const noexcept override;

    /// @brief 返回参数集合指针
    /// @return 
    PluginFieldCollection const* getFieldNames() noexcept override;

    /// @brief 动态创建插件实例，解析 fc 中的参数（如 iou_threshold），填充到 mParam
    /// @param name 插件名称
    /// @param fc   包含算子属性的结构体
    /// @return 
    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    
    /// @brief 将序列化数据反序列化为可执行的插件实例，从 serialData 中读取序列化参数（如 iou_threshold）
    /// @param name 插件实例名称
    /// @param serialData 指向序列化数据的指针
    /// @param serialLength 序列化数据的字节长度
    /// @return 
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

protected:
    PluginFieldCollection mFC;                      ///< 存储插件参数的集合
    EfficientRotatedNMSParameters mParam;           ///< 存储插件核心参数
    std::vector<PluginField> mPluginAttributes;     ///< 存放插件参数的中间结构，用于初始化 mFC
    std::string mPluginName;                        ///< 插件名称
};

} // namespace plugin
} // namespace nvinfer1

#endif // TRT_EFFICIENT_ROTATED_NMS_PLUGIN_H
