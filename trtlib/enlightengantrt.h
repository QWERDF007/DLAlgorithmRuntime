#pragma once

#include <opencv2/opencv.hpp>

#include <NvInfer.h>


class __declspec(dllexport) EnlightenGAN {
public:
    EnlightenGAN(const std::string &model_path);

    ~EnlightenGAN();

    void predict(cv::Mat image, cv::OutputArray enlighted);

private:
    nvinfer1::IRuntime *runtime_{ nullptr };
    nvinfer1::ICudaEngine *engine_{ nullptr };
    nvinfer1::IExecutionContext *context_{ nullptr };
    int input1_index_, input2_index_;
    int output_index_, latent_index_;
    size_t input1_size_, input2_size_, output_size_, latent_size_;
};