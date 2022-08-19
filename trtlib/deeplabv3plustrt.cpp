#include "DeepLabv3Plus.h"
#include <fstream>
#include <iostream>
#include <vector>


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // Only log Warnings or more important.
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};


DeepLabv3Plus::DeepLabv3Plus(const std::string &model_path) {
    //std::cout << "mean: " << mean << " std: " << std << std::endl;
    Logger logger;
    std::fstream file;
    std::cout << "Loading file from: " << model_path << std::endl;
    file.open(model_path, std::ios::binary | std::ios::in);
    file.seekg(0, std::ios::end);
    int size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        file.close();
        throw std::runtime_error("Unable to read engine file!");
    }
    file.close();
    std::cout << "Creating Infer Runtime" << std::endl;
    runtime_ = nvinfer1::createInferRuntime(logger);
    if (!runtime_) {
        throw std::runtime_error("Unable to create infer runtime!");
    }
    std::cout << "Deserializing Cuda Engine" << std::endl;

    engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        throw std::runtime_error("Unable to deserialize cuda engine!");
    }
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    std::cout << "Done" << std::endl;

    input_index_ = engine_->getBindingIndex("input");
    output_index_ = engine_->getBindingIndex("output");

    auto dims1 = engine_->getBindingDimensions(input_index_);
    auto dims2 = engine_->getBindingDimensions(output_index_);

    std::cout << dims1.d[0] << " " << dims1.d[1] << " " << dims1.d[2] << " " << dims1.d[3] << std::endl;
    std::cout << dims2.d[0] << " " << dims2.d[1] << " " << dims2.d[2] << " " << dims2.d[3] << std::endl;

    input_size_ = dims1.d[0] * dims1.d[1] * dims1.d[2] * dims1.d[3];
    output_size_ = dims2.d[0] * dims2.d[1] * dims2.d[2] * dims2.d[3];
    num_classes_ = dims2.d[1];
}

DeepLabv3Plus::~DeepLabv3Plus() {
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
}

void DeepLabv3Plus::predict(cv::Mat image, cv::OutputArray segment) {
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1, 0);
    rgb = (rgb - cv::Scalar(123.675, 116.28, 103.53)) / cv::Scalar(58.395, 57.12, 57.375);
    const int height = rgb.rows;
    const int width = rgb.cols;

    cv::Mat blob_image = cv::dnn::blobFromImage(rgb);
    void *buffers[2];
    int32_t *outdata = new int32_t[output_size_];

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&buffers[input_index_], input_size_ * sizeof(float));
    cudaMalloc(&buffers[output_index_], output_size_ * sizeof(int32_t));

    cudaMemcpyAsync(buffers[input_index_], blob_image.data, input_size_ * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_->enqueue(1, buffers, stream, nullptr);
    cudaMemcpyAsync(outdata, buffers[output_index_], output_size_ * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);


    segment.create(height, width, CV_8UC1);
    cv::Mat _segment = segment.getMat(-1);

    size_t i = 0;
    for (int h = 0; h < height; ++h) {
        uchar *ptr = _segment.ptr<uchar>(h);
        for (int w = 0; w < width; ++w) {
            ptr[w] = static_cast<uchar>(outdata[i++]);
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[input_index_]);
    cudaFree(buffers[output_index_]);
    
    delete[] outdata;
}

