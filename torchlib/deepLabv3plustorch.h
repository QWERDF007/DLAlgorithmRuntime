#pragma once

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <opencv2/opencv.hpp>


class __declspec(dllexport) DeepLabv3Plus {
public:
    DeepLabv3Plus(std::string &model_path, int device_id = 0);

    ~DeepLabv3Plus();

    void predict(cv::Mat image, cv::OutputArray segment);

    void setThreads(int threads);

    int threads();

private:
    torch::jit::script::Module module_;

    torch::Device device_;

    void mat2Tensor(const cv::Mat &imageFloat, torch::Tensor *outTensor);

    void tensor2Mat(const torch::Tensor &inputTensor, cv::Mat *imageFloat);
};