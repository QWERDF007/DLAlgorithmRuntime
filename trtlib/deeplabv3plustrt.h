#pragma once

#include <opencv2/opencv.hpp>

#include <NvInfer.h>


class __declspec(dllexport) DeepLabv3Plus {
public:
    DeepLabv3Plus(const std::string &model_path);

    ~DeepLabv3Plus();

    void predict(cv::Mat image, cv::OutputArray segment);

private:
    nvinfer1::IRuntime *runtime_{ nullptr };
    nvinfer1::ICudaEngine *engine_{ nullptr };
    nvinfer1::IExecutionContext *context_{ nullptr };
    int input_index_;
    int output_index_;
    int num_classes_;
    size_t input_size_, output_size_;
    //cv::Scalar mean{ 123.675, 116.28, 103.53 };
    //cv::Scalar std{ 58.395, 57.12, 57.375 };
};