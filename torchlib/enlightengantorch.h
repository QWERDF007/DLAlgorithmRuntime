#pragma once

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <opencv2/opencv.hpp>


class __declspec(dllexport) EnlightenGAN {
public:
    EnlightenGAN(std::string model, int device_id = 0);

    ~EnlightenGAN();

    void predict(cv::Mat image, cv::OutputArray enlighted);

    void setThreads(int threads);

    int threads();

private:
    torch::jit::script::Module _module;

    torch::Device _device;

    void mat2Tensor(const cv::Mat &imageFloat, torch::Tensor *outTensor);

    void tensor2Mat(const torch::Tensor &inputTensor, cv::Mat *imageFloat);
};