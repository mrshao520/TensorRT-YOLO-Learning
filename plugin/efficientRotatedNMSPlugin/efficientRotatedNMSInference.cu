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

#include "common/bboxUtils.h"
#include "cub/cub.cuh"
#include "cuda_runtime_api.h"

#include "efficientRotatedNMSInference.cuh"
#include "efficientRotatedNMSInference.h"

#define NMS_TILES 5

using namespace nvinfer1;
using namespace nvinfer1::plugin;

template <typename T>
__device__ float IOU(EfficientRotatedNMSParameters param, RotatedBoxCorner<T> box1, RotatedBoxCorner<T> box2)
{
    // Regardless of the selected box coding, IOU is always performed in RotatedBoxCorner coding.
    // The boxes are copied so that they can be reordered without affecting the originals.
    RotatedBoxCorner<T> b1 = box1;
    RotatedBoxCorner<T> b2 = box2;
    b1.reorder();
    b2.reorder();
    return RotatedBoxCorner<T>::probiou(b1, b2);
}

template <typename T, typename Tb>
__device__ RotatedBoxCorner<T> DecodeBoxes(EfficientRotatedNMSParameters param, int boxIdx, int anchorIdx,
    const Tb* __restrict__ boxesInput, const Tb* __restrict__ anchorsInput)
{
    // The inputs will be in the selected coding format, as well as the decoding function. But the decoded box
    // will always be returned as RotatedBoxCorner.
    Tb box = boxesInput[boxIdx];
    if (!param.boxDecoder)
    {
        return RotatedBoxCorner<T>(box);
    }
    Tb anchor = anchorsInput[anchorIdx];
    box.reorder();
    anchor.reorder();
    return RotatedBoxCorner<T>(box.decode(anchor));
}

template <typename T, typename Tb>
__device__ void MapRotatedNMSData(EfficientRotatedNMSParameters param, int idx, int imageIdx, const Tb* __restrict__ boxesInput,
    const Tb* __restrict__ anchorsInput, const int* __restrict__ topClassData, const int* __restrict__ topAnchorsData,
    const int* __restrict__ topNumData, const T* __restrict__ sortedScoresData, const int* __restrict__ sortedIndexData,
    T& scoreMap, int& classMap, RotatedBoxCorner<T>& boxMap, int& boxIdxMap)
{
    // idx: Holds the NMS box index, within the current batch.
    // idxSort: Holds the batched NMS box index, which indexes the (filtered, but sorted) score buffer.
    // scoreMap: Holds the score that corresponds to the indexed box being processed by NMS.
    if (idx >= topNumData[imageIdx])
    {
        return;
    }
    int idxSort = imageIdx * param.numScoreElements + idx;
    scoreMap = sortedScoresData[idxSort];

    // idxMap: Holds the re-mapped index, which indexes the (filtered, but unsorted) buffers.
    // classMap: Holds the class that corresponds to the idx'th sorted score being processed by NMS.
    // anchorMap: Holds the anchor that corresponds to the idx'th sorted score being processed by NMS.
    int idxMap = imageIdx * param.numScoreElements + sortedIndexData[idxSort];
    classMap = topClassData[idxMap];
    int anchorMap = topAnchorsData[idxMap];

    // boxIdxMap: Holds the re-re-mapped index, which indexes the (unfiltered, and unsorted) boxes input buffer.
    boxIdxMap = -1;
    if (param.shareLocation) // Shape of boxesInput: [batchSize, numAnchors, 1, 4]
    {
        boxIdxMap = imageIdx * param.numAnchors + anchorMap;
    }
    else // Shape of boxesInput: [batchSize, numAnchors, numClasses, 4]
    {
        int batchOffset = imageIdx * param.numAnchors * param.numClasses;
        int anchorOffset = anchorMap * param.numClasses;
        boxIdxMap = batchOffset + anchorOffset + classMap;
    }
    // anchorIdxMap: Holds the re-re-mapped index, which indexes the (unfiltered, and unsorted) anchors input buffer.
    int anchorIdxMap = -1;
    if (param.shareAnchors) // Shape of anchorsInput: [1, numAnchors, 4]
    {
        anchorIdxMap = anchorMap;
    }
    else // Shape of anchorsInput: [batchSize, numAnchors, 4]
    {
        anchorIdxMap = imageIdx * param.numAnchors + anchorMap;
    }
    // boxMap: Holds the box that corresponds to the idx'th sorted score being processed by NMS.
    boxMap = DecodeBoxes<T, Tb>(param, boxIdxMap, anchorIdxMap, boxesInput, anchorsInput);
}

template <typename T>
__device__ void WriteRotatedNMSResult(EfficientRotatedNMSParameters param, int* __restrict__ numDetectionsOutput,
    T* __restrict__ nmsScoresOutput, int* __restrict__ nmsClassesOutput, RotatedBoxCorner<T>* __restrict__ nmsBoxesOutput,
    T threadScore, int threadClass, RotatedBoxCorner<T> threadBox, int imageIdx, unsigned int resultsCounter)
{
    int outputIdx = imageIdx * param.numOutputBoxes + resultsCounter - 1;
    if (param.scoreSigmoid)
    {
        nmsScoresOutput[outputIdx] = sigmoid_mp(threadScore);
    }
    else if (param.scoreBits > 0)
    {
        nmsScoresOutput[outputIdx] = add_mp(threadScore, (T) -1);
    }
    else
    {
        nmsScoresOutput[outputIdx] = threadScore;
    }
    nmsClassesOutput[outputIdx] = threadClass;
    if (param.clipBoxes)
    {
        nmsBoxesOutput[outputIdx] = threadBox.clip((T) 0, (T) 1);
    }
    else
    {
        nmsBoxesOutput[outputIdx] = threadBox;
    }
    numDetectionsOutput[imageIdx] = resultsCounter;
}

template <typename T, typename Tb>
__global__ void EfficientRotatedNMS(EfficientRotatedNMSParameters param, const int* topNumData, int* outputIndexData,
    int* outputClassData, const int* sortedIndexData, const T* __restrict__ sortedScoresData,
    const int* __restrict__ topClassData, const int* __restrict__ topAnchorsData, const Tb* __restrict__ boxesInput,
    const Tb* __restrict__ anchorsInput, int* __restrict__ numDetectionsOutput, T* __restrict__ nmsScoresOutput,
    int* __restrict__ nmsClassesOutput, RotatedBoxCorner<T>* __restrict__ nmsBoxesOutput)
{
    unsigned int thread = threadIdx.x;  ///< 线程 id
    unsigned int imageIdx = blockIdx.y; ///< 图像批次索引
    unsigned int tileSize = blockDim.x; ///< 线程块大小

    /// 检查图像索引有效性
    if (imageIdx >= param.batchSize)
    {
        return;
    }

    /// 确定实际处理的候选框数量
    int numSelectedBoxes = min(topNumData[imageIdx], param.numSelectedBoxes);
    /// 计算需要的分块数
    int numTiles = (numSelectedBoxes + tileSize - 1) / tileSize;
    /// 检查线程有效性
    if (thread >= numSelectedBoxes)
    {
        return;
    }

    __shared__ int blockState; ///< 块状态（线程间通信）
    __shared__ unsigned int resultsCounter; ///< 结果计数器
    if (thread == 0)
    {
        blockState = 0; ///< 初始状态; 0=正常
        resultsCounter = 0; ///< 结果计数清零
    }

    int threadState[NMS_TILES]; ///< 框状态
    unsigned int boxIdx[NMS_TILES]; ///< 框全局索引
    T threadScore[NMS_TILES]; ///< 框得分
    int threadClass[NMS_TILES]; ///< 框类别
    RotatedBoxCorner<T> threadBox[NMS_TILES]; ///< 旋转框索引
    int boxIdxMap[NMS_TILES]; ///< 框映射索引

    /// 初始化并加载数据
    for (int tile = 0; tile < numTiles; tile++)
    {
        threadState[tile] = 0; ///< 初始状态
        boxIdx[tile] = thread + tile * blockDim.x; ///< 计算全局索引
        /// 加载框数据到线程私有存储
        MapRotatedNMSData<T, Tb>(param, boxIdx[tile], imageIdx, boxesInput, anchorsInput, topClassData, topAnchorsData,
            topNumData, sortedScoresData, sortedIndexData, threadScore[tile], threadClass[tile], threadBox[tile],
            boxIdxMap[tile]);
    }

    // Iterate through all boxes to NMS against.
    for (int i = 0; i < numSelectedBoxes; i++)
    {
        int tile = i / tileSize; /// 当前框所属分块
        
        if (boxIdx[tile] == i)
        {
            // Iteration lead thread, figure out what the other threads should do,
            // this will be signaled via the blockState shared variable.
            if (threadState[tile] == -1)
            {
                // Thread already dead, this box was already dropped in a previous iteration,
                // because it had a large IOU overlap with another lead thread previously, so
                // it would never be kept anyway, therefore it can safely be skip all IOU operations
                // in this iteration.
                /// 状态：跳过迭代，框已被丢弃
                blockState = -1; // -1 => Signal all threads to skip iteration
            }
            else if (threadState[tile] == 0)
            {
                // As this box will be kept, this is a good place to find what index in the results buffer it
                // should have, as this allows to perform an early loop exit if there are enough results.
                if (resultsCounter >= param.numOutputBoxes)
                {
                    /// 状态：提前退出，结果数已达上限
                    blockState = -2; // -2 => Signal all threads to do an early loop exit.
                }
                else
                {
                    // Thread is still alive, because it has not had a large enough IOU overlap with
                    // any other kept box previously. Therefore, this box will be kept for sure. However,
                    // we need to check against all other subsequent boxes from this position onward,
                    // to see how those other boxes will behave in future iterations.
                    blockState = 1;        // +1 => Signal all (higher index) threads to calculate IOU against this box
                    threadState[tile] = 1; // +1 => Mark this box's thread to be kept and written out to results

                    // If the numOutputBoxesPerClass check is enabled, write the result only if the limit for this
                    // class on this image has not been reached yet. Other than (possibly) skipping the write, this
                    // won't affect anything else in the NMS threading.
                    bool write = true;
                    if (param.numOutputBoxesPerClass >= 0)
                    {
                        int classCounterIdx = imageIdx * param.numClasses + threadClass[tile];
                        write = (outputClassData[classCounterIdx] < param.numOutputBoxesPerClass);
                        outputClassData[classCounterIdx]++;
                    }
                    if (write)
                    {
                        // This branch is visited by one thread per iteration, so it's safe to do non-atomic increments.
                        resultsCounter++;
                        WriteRotatedNMSResult<T>(param, numDetectionsOutput, nmsScoresOutput, nmsClassesOutput,
                            nmsBoxesOutput, threadScore[tile], threadClass[tile], threadBox[tile], imageIdx,
                            resultsCounter);
                    }
                }
            }
            else
            {
                // This state should never be reached, but just in case...
                blockState = 0; // 0 => Signal all threads to not do any updates, nothing happens.
            }
        }

        __syncthreads();

        if (blockState == -2)
        {
            // This is the signal to exit from the loop.
            return; ///< 提取推出
        }

        if (blockState == -1)
        {
            // This is the signal for all threads to just skip this iteration, as no IOU's need to be checked.
            continue; ///< 跳过本次迭代
        }

        // Grab a box and class to test the current box against. The test box corresponds to iteration i,
        // therefore it will have a lower index than the current thread box, and will therefore have a higher score
        // than the current box because it's located "before" in the sorted score list.
        T testScore;
        int testClass;
        RotatedBoxCorner<T> testBox;
        int testBoxIdxMap;

        /// 获取测试框数据
        MapRotatedNMSData<T, Tb>(param, i, imageIdx, boxesInput, anchorsInput, topClassData, topAnchorsData, topNumData,
            sortedScoresData, sortedIndexData, testScore, testClass, testBox, testBoxIdxMap);

        for (int tile = 0; tile < numTiles; tile++)
        {
            bool ignoreClass = true;
            if (!param.classAgnostic)
            {
                ignoreClass = threadClass[tile] == testClass;
            }

            // IOU
            if (boxIdx[tile] > i && // 只处理索引更大的框 Make sure two different boxes are being tested, and that it's a higher index;
                boxIdx[tile] < numSelectedBoxes && // 索引有效 Make sure the box is within numSelectedBoxes;
                blockState == 1 &&                 // 状态允许计算 Signal that allows IOU checks to be performed;
                threadState[tile] == 0 &&          // 框状态为待处理 Make sure this box hasn't been either dropped or kept already;
                ignoreClass &&                     // 类别无关 Compare only boxes of matching classes when classAgnostic is false;
                lte_mp(threadScore[tile], testScore) && // 得分排序 Make sure the sorting order of scores is as expected;
                IOU<T>(param, threadBox[tile], testBox) >= param.iouThreshold) // IOU阈值判断 And... IOU overlap.
            {
                // Current box overlaps with the box tested in this iteration, this box will be skipped.
                threadState[tile] = -1; // -1 => Mark this box's thread to be dropped.
            }
        }
    }
}

template <typename T>
cudaError_t EfficientRotatedNMSLauncher(EfficientRotatedNMSParameters& param, int* topNumData, int* outputIndexData,
    int* outputClassData, int* sortedIndexData, T* sortedScoresData, int* topClassData, int* topAnchorsData,
    const void* boxesInput, const void* anchorsInput, int* numDetectionsOutput, T* nmsScoresOutput,
    int* nmsClassesOutput, void* nmsBoxesOutput, cudaStream_t stream)
{
    /// 根据候选框数量调整线程块大小，优化并行效率
    unsigned int tileSize = param.numSelectedBoxes / NMS_TILES;
    if (param.numSelectedBoxes <= 512)
    {
        tileSize = 512;
    }
    if (param.numSelectedBoxes <= 256)
    {
        tileSize = 256;
    }

    /// 每个线程块处理一个图像样本
    const dim3 blockSize = {tileSize, 1, 1};  ///< 一维线程块
    const dim3 gridSize = {1, (unsigned int) param.batchSize, 1}; ///< 二维网格

    if (param.boxCoding == 0)
    {
        /// 角点编码：x1 y1 x2 y2 r
        EfficientRotatedNMS<T, RotatedBoxCorner<T>><<<gridSize, blockSize, 0, stream>>>(param, topNumData, outputIndexData,
            outputClassData, sortedIndexData, sortedScoresData, topClassData, topAnchorsData,
            (RotatedBoxCorner<T>*) boxesInput, (RotatedBoxCorner<T>*) anchorsInput, numDetectionsOutput, nmsScoresOutput,
            nmsClassesOutput, (RotatedBoxCorner<T>*) nmsBoxesOutput);
    }
    else if (param.boxCoding == 1)
    {
        /// 中心编码：x y w h r
        // Note that nmsBoxesOutput is always coded as RotatedBoxCorner<T>, regardless of the input coding type.
        EfficientRotatedNMS<T, RotatedBoxCenterSize<T>><<<gridSize, blockSize, 0, stream>>>(param, topNumData, outputIndexData,
            outputClassData, sortedIndexData, sortedScoresData, topClassData, topAnchorsData,
            (RotatedBoxCenterSize<T>*) boxesInput, (RotatedBoxCenterSize<T>*) anchorsInput, numDetectionsOutput, nmsScoresOutput,
            nmsClassesOutput, (RotatedBoxCorner<T>*) nmsBoxesOutput);
    }

    return cudaGetLastError();
}

__global__ void EfficientRotatedNMSFilterSegments(EfficientRotatedNMSParameters param, const int* __restrict__ topNumData,
    int* __restrict__ topOffsetsStartData, int* __restrict__ topOffsetsEndData)
{
    /// 每个线程处理一个图像样本 <<<1, param.batchSize, 0, stream>>>
    int imageIdx = threadIdx.x;
    if (imageIdx > param.batchSize)
    {
        return;
    }
    /// 当前图像在全局数组中的起始索引 图像索引*最大候选框数
    topOffsetsStartData[imageIdx] = imageIdx * param.numScoreElements;
    /// 当前图像在全局数组中的结束索引 起始索引+当前图像实际候选框数
    topOffsetsEndData[imageIdx] = imageIdx * param.numScoreElements + topNumData[imageIdx];
}

template <typename T>
__global__ void EfficientRotatedNMSFilter(EfficientRotatedNMSParameters param, const T* __restrict__ scoresInput,
    int* __restrict__ topNumData, int* __restrict__ topIndexData, int* __restrict__ topAnchorsData,
    T* __restrict__ topScoresData, int* __restrict__ topClassData)
{
    /// 候选框元素索引
    int elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
    /// 图像批次索引
    int imageIdx = blockDim.y * blockIdx.y + threadIdx.y;

    // Boundary Conditions 边界检查
    if (elementIdx >= param.numScoreElements || imageIdx >= param.batchSize)
    {
        return;
    }

    // Shape of scoresInput: [batchSize, numAnchors, numClasses]
    /// 获取 num_classes 索引
    int scoresInputIdx = imageIdx * param.numScoreElements + elementIdx;

    // For each class, check its corresponding score if it crosses the threshold, and if so select this anchor,
    // and keep track of the maximum score and the corresponding (argmax) class id
    T score = scoresInput[scoresInputIdx];
    if (gte_mp(score, (T) param.scoreThreshold))
    {
        // Unpack the class and anchor index from the element index
        int classIdx = elementIdx % param.numClasses;  ///< 类别索引
        int anchorIdx = elementIdx / param.numClasses; ///< 目标框索引

        // If this is a background class, ignore it. 背景过滤
        if (classIdx == param.backgroundClass)
        {
            return;
        }

        // Use an atomic to find an open slot where to write the selected anchor data.
        /// 预检查：避免已满图像的无效原子操作
        if (topNumData[imageIdx] >= param.numScoreElements)
        {
            return;
        }
        int selectedIdx = atomicAdd((unsigned int*) &topNumData[imageIdx], 1);
        /// 后检查：确保写入位置不越界
        if (selectedIdx >= param.numScoreElements)
        {
            topNumData[imageIdx] = param.numScoreElements;
            return;
        }

        // Shape of topScoresData / topClassData: [batchSize, numScoreElements]
        int topIdx = imageIdx * param.numScoreElements + selectedIdx;

        /// 为后续的浮点数位排序做准备
        if (param.scoreBits > 0)
        {
            score = add_mp(score, (T) 1);
            if (gt_mp(score, (T) (2.f - 1.f / 1024.f)))
            {
                // Ensure the incremented score fits in the mantissa without changing the exponent
                score = (2.f - 1.f / 1024.f);
            }
        }

        topIndexData[topIdx] = selectedIdx; ///< 输出缓冲区索引
        topAnchorsData[topIdx] = anchorIdx; ///< 目标狂索引
        topScoresData[topIdx] = score;      ///< 优化后的分数
        topClassData[topIdx] = classIdx;    ///< 类别索引
    }
}

template <typename T>
__global__ void EfficientRotatedNMSDenseIndex(EfficientRotatedNMSParameters param, int* __restrict__ topNumData,
    int* __restrict__ topIndexData, int* __restrict__ topAnchorsData, int* __restrict__ topOffsetsStartData,
    int* __restrict__ topOffsetsEndData, T* __restrict__ topScoresData, int* __restrict__ topClassData)
{
    int elementIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int imageIdx = blockDim.y * blockIdx.y + threadIdx.y;

    if (elementIdx >= param.numScoreElements || imageIdx >= param.batchSize)
    {
        return;
    }

    int dataIdx = imageIdx * param.numScoreElements + elementIdx;
    int anchorIdx = elementIdx / param.numClasses;
    int classIdx = elementIdx % param.numClasses;
    if (param.scoreBits > 0)
    {
        T score = topScoresData[dataIdx];
        if (lt_mp(score, (T) param.scoreThreshold))
        {
            score = (T) 1;
        }
        else if (classIdx == param.backgroundClass)
        {
            score = (T) 1;
        }
        else
        {
            score = add_mp(score, (T) 1);
            if (gt_mp(score, (T) (2.f - 1.f / 1024.f)))
            {
                // Ensure the incremented score fits in the mantissa without changing the exponent
                score = (2.f - 1.f / 1024.f);
            }
        }
        topScoresData[dataIdx] = score;
    }
    else
    {
        T score = topScoresData[dataIdx];
        if (lt_mp(score, (T) param.scoreThreshold))
        {
            topScoresData[dataIdx] = -(1 << 15);
        }
        else if (classIdx == param.backgroundClass)
        {
            topScoresData[dataIdx] = -(1 << 15);
        }
    }

    topIndexData[dataIdx] = elementIdx;
    topAnchorsData[dataIdx] = anchorIdx;
    topClassData[dataIdx] = classIdx;

    if (elementIdx == 0)
    {
        // Saturate counters
        topNumData[imageIdx] = param.numScoreElements;
        topOffsetsStartData[imageIdx] = imageIdx * param.numScoreElements;
        topOffsetsEndData[imageIdx] = (imageIdx + 1) * param.numScoreElements;
    }
}

/// @brief 筛选出符合阈值要求的候选框，并为后续排序和NMS准备数据
/// @tparam T 
/// @param param 参数列表
/// @param scoresInput 原始 scores 数据
/// @param topNumData 每个样本保留的候选框数量
/// @param topIndexData 保留候选框的索引
/// @param topAnchorsData 保留候选框对应的锚点索引
/// @param topOffsetsStartData 分段排序的起始索引
/// @param topOffsetsEndData 分段排序的结束索引
/// @param topScoresData 过滤后的得分
/// @param topClassData 候选框类别ID
/// @param stream 
/// @return 
template <typename T>
cudaError_t EfficientRotatedNMSFilterLauncher(EfficientRotatedNMSParameters& param, const T* scoresInput, int* topNumData,
    int* topIndexData, int* topAnchorsData, int* topOffsetsStartData, int* topOffsetsEndData, T* topScoresData,
    int* topClassData, cudaStream_t stream)
{
    /// X 维度：按候选框元素并行 每块 512 个元素
    /// Y 维度：按图像批次并行 每块 1 个样本
    const unsigned int elementsPerBlock = 512;
    const unsigned int imagesPerBlock = 1;
    const unsigned int elementBlocks = (param.numScoreElements + elementsPerBlock - 1) / elementsPerBlock;
    const unsigned int imageBlocks = (param.batchSize + imagesPerBlock - 1) / imagesPerBlock;
    const dim3 blockSize = {elementsPerBlock, imagesPerBlock, 1};
    const dim3 gridSize = {elementBlocks, imageBlocks, 1};

    /// 将阈值反向计算得到 Logit 空间（Sigmoid 反函数）
    float kernelSelectThreshold = 0.007f;
    if (param.scoreSigmoid)
    {
        // Inverse Sigmoid
        if (param.scoreThreshold <= 0.f)
        {
            param.scoreThreshold = -(1 << 15);
        }
        else
        {
            param.scoreThreshold = logf(param.scoreThreshold / (1.f - param.scoreThreshold));
        }
        kernelSelectThreshold = logf(kernelSelectThreshold / (1.f - kernelSelectThreshold));
        // Disable Score Bits Optimization
        param.scoreBits = -1;
    }

    /// 根据阈值选择高效执行路径
    if (param.scoreThreshold < kernelSelectThreshold)
    {
        /**
         * 低阈值路径，保留大部分候选框
         * 避免条件分支，用内存复制换取计算效率
        */
        // A full copy of the buffer is necessary because sorting will scramble the input data otherwise.
        /// 直接复制全部score
        PLUGIN_CHECK_CUDA(cudaMemcpyAsync(topScoresData, scoresInput,
            param.batchSize * param.numScoreElements * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        /// 生成密集索引
        EfficientRotatedNMSDenseIndex<T><<<gridSize, blockSize, 0, stream>>>(param, topNumData, topIndexData, topAnchorsData,
            topOffsetsStartData, topOffsetsEndData, topScoresData, topClassData);
    }
    else
    {
        /// 通过原子操作压缩有效数据，减少后续处理量
        EfficientRotatedNMSFilter<T><<<gridSize, blockSize, 0, stream>>>(
            param, scoresInput, topNumData, topIndexData, topAnchorsData, topScoresData, topClassData);

        /// 计算每个图像（在批次中）在过滤后的候选框的起始和结束偏移量
        EfficientRotatedNMSFilterSegments<<<1, param.batchSize, 0, stream>>>(
            param, topNumData, topOffsetsStartData, topOffsetsEndData);
    }

    return cudaGetLastError();
}

template <typename T>
size_t EfficientRotatedNMSSortWorkspaceSize(int batchSize, int numScoreElements)
{
    size_t sortedWorkspaceSize = 0;
    cub::DoubleBuffer<T> keysDB(nullptr, nullptr);
    cub::DoubleBuffer<int> valuesDB(nullptr, nullptr);
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, sortedWorkspaceSize, keysDB, valuesDB,
        numScoreElements, batchSize, (const int*) nullptr, (const int*) nullptr);
    return sortedWorkspaceSize;
}

size_t EfficientRotatedNMSWorkspaceSize(int batchSize, int numScoreElements, int numClasses, DataType datatype)
{
    size_t total = 0;
    const size_t align = 256;
    // Counters
    // 3 for Filtering
    // 1 for Output Indexing
    // C for Max per Class Limiting
    size_t size = (3 + 1 + numClasses) * batchSize * sizeof(int);
    total += size + (size % align ? align - (size % align) : 0);
    // Int Buffers
    for (int i = 0; i < 4; i++)
    {
        size = batchSize * numScoreElements * sizeof(int);
        total += size + (size % align ? align - (size % align) : 0);
    }
    // Float Buffers
    for (int i = 0; i < 2; i++)
    {
        size = batchSize * numScoreElements * dataTypeSize(datatype);
        total += size + (size % align ? align - (size % align) : 0);
    }
    // Sort Workspace
    if (datatype == DataType::kHALF)
    {
        size = EfficientRotatedNMSSortWorkspaceSize<__half>(batchSize, numScoreElements);
        total += size + (size % align ? align - (size % align) : 0);
    }
    else if (datatype == DataType::kFLOAT)
    {
        size = EfficientRotatedNMSSortWorkspaceSize<float>(batchSize, numScoreElements);
        total += size + (size % align ? align - (size % align) : 0);
    }

    return total;
}

template <typename T>
T* EfficientRotatedNMSWorkspace(void* workspace, size_t& offset, size_t elements)
{
    T* buffer = (T*) ((size_t) workspace + offset);
    size_t align = 256;
    size_t size = elements * sizeof(T);
    size_t sizeAligned = size + (size % align ? align - (size % align) : 0);
    offset += sizeAligned;
    return buffer;
}

template <typename T>
pluginStatus_t EfficientRotatedNMSDispatch(EfficientRotatedNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* workspace, cudaStream_t stream)
{
    // Clear Outputs (not all elements will get overwritten by the kernels, so safer to clear everything out)
    /// 1. 将所有输出缓冲区初始化为0，确保未处理的元素不会残留无效数据
    CSC(cudaMemsetAsync(numDetectionsOutput, 0x00, param.batchSize * sizeof(int), stream), STATUS_FAILURE);
    CSC(cudaMemsetAsync(nmsScoresOutput, 0x00, param.batchSize * param.numOutputBoxes * sizeof(T), stream), STATUS_FAILURE);
    CSC(cudaMemsetAsync(nmsBoxesOutput, 0x00, param.batchSize * param.numOutputBoxes * 5 * sizeof(T), stream), STATUS_FAILURE);
    CSC(cudaMemsetAsync(nmsClassesOutput, 0x00, param.batchSize * param.numOutputBoxes * sizeof(int), stream), STATUS_FAILURE);

    // Empty Inputs
    /// 2. 若输入得分元素数量为0（无有效检测），直接返回成功
    if (param.numScoreElements < 1)
    {
        return STATUS_SUCCESS;
    }

    // Counters Workspace
    /// 3. 分配临时内存用于中间计算结果
    size_t workspaceOffset = 0; ///< 计算workspace后，offset会移动
    int countersTotalSize = (3 + 1 + param.numClasses) * param.batchSize;
    /// 每个batch保留的检测框数量
    int* topNumData = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, countersTotalSize);
    /// 记录每个batch在排序后的起始和结束索引
    int* topOffsetsStartData = topNumData + param.batchSize;
    int* topOffsetsEndData = topNumData + 2 * param.batchSize;
    /// 存储最终输出的索引和类别信息
    int* outputIndexData = topNumData + 3 * param.batchSize;
    int* outputClassData = topNumData + 4 * param.batchSize;
    CSC(cudaMemsetAsync(topNumData, 0x00, countersTotalSize * sizeof(int), stream), STATUS_FAILURE);
    cudaError_t status = cudaGetLastError();
    CSC(status, STATUS_FAILURE);

    // Other Buffers Workspace
    /// 4. 分配排序与过滤缓冲区
    int* topIndexData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* topClassData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* topAnchorsData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    int* sortedIndexData
        = EfficientRotatedNMSWorkspace<int>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    T* topScoresData = EfficientRotatedNMSWorkspace<T>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    T* sortedScoresData
        = EfficientRotatedNMSWorkspace<T>(workspace, workspaceOffset, param.batchSize * param.numScoreElements);
    size_t sortedWorkspaceSize = EfficientRotatedNMSSortWorkspaceSize<T>(param.batchSize, param.numScoreElements);
    char* sortedWorkspaceData = EfficientRotatedNMSWorkspace<char>(workspace, workspaceOffset, sortedWorkspaceSize);
    cub::DoubleBuffer<T> scoresDB(topScoresData, sortedScoresData);
    cub::DoubleBuffer<int> indexDB(topIndexData, sortedIndexData);

    // Kernels 核心处理流程

    /// 5.1 过滤低分候选框
    status = EfficientRotatedNMSFilterLauncher<T>(param, (T*) scoresInput, topNumData, topIndexData, topAnchorsData,
        topOffsetsStartData, topOffsetsEndData, topScoresData, topClassData, stream);
    CSC(status, STATUS_FAILURE);
    
    /// 5.2 分段排序
    status = cub::DeviceSegmentedRadixSort::SortPairsDescending(sortedWorkspaceData, sortedWorkspaceSize, scoresDB,
        indexDB, param.batchSize * param.numScoreElements, param.batchSize, topOffsetsStartData, topOffsetsEndData,
        param.scoreBits > 0 ? (10 - param.scoreBits) : 0, param.scoreBits > 0 ? 10 : sizeof(T) * 8, stream);
    CSC(status, STATUS_FAILURE);

    /// 5.3 执行旋转NMS
    status = EfficientRotatedNMSLauncher<T>(param, topNumData, outputIndexData, outputClassData, indexDB.Current(),
        scoresDB.Current(), topClassData, topAnchorsData, boxesInput, anchorsInput, (int*) numDetectionsOutput,
        (T*) nmsScoresOutput, (int*) nmsClassesOutput, nmsBoxesOutput, stream);
    CSC(status, STATUS_FAILURE);

    return STATUS_SUCCESS;
}

pluginStatus_t EfficientRotatedNMSInference(EfficientRotatedNMSParameters param, const void* boxesInput, const void* scoresInput,
    const void* anchorsInput, void* numDetectionsOutput, void* nmsBoxesOutput, void* nmsScoresOutput,
    void* nmsClassesOutput, void* workspace, cudaStream_t stream)
{
    if (param.datatype == DataType::kFLOAT)
    {
        /// FP32 通用推理
        param.scoreBits = -1;
        return EfficientRotatedNMSDispatch<float>(param, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, workspace, stream);
    }
    else if (param.datatype == DataType::kHALF)
    {
        /// FP16 推理
        if (param.scoreBits <= 0 || param.scoreBits > 10)
        {
            param.scoreBits = -1;
        }
        return EfficientRotatedNMSDispatch<__half>(param, boxesInput, scoresInput, anchorsInput, numDetectionsOutput,
            nmsBoxesOutput, nmsScoresOutput, nmsClassesOutput, workspace, stream);
    }
    else
    {
        return STATUS_NOT_SUPPORTED;
    }
}
