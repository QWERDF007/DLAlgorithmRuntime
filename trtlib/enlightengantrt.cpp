#include "EnlightenGAN.h"
#include "EnlightenGAN.h"
#include <fstream>
#include <iostream>
#include <vector>

#include <NvOnnxParser.h>


class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *msg) noexcept override {
        // Only log Warnings or more important.
        if (severity <= Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
};

auto logger = Logger();

EnlightenGAN::EnlightenGAN(const std::string &model_path) {
    //auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    //uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    //auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    //auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));

    //std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    //std::streamsize size = file.tellg();
    //file.seekg(0, std::ios::beg);
    //
    //std::vector<char> buffer(size);
    //if (!file.read(buffer.data(), size)) {
    //    throw std::runtime_error("Unable to read engine file");
    //}

    //auto parsed = parser->parse(buffer.data(), buffer.size());
    //if (!parsed) {
    //    throw std::runtime_error("Unable to parse engine file");
    //}

    //// Save the input height, width, and channels.
    //// Require this info for inference.
    //const auto input1 = network->getInput(0);
    //const auto input2 = network->getInput(1);
    //const auto output = network->getOutput(0);
    //const auto inputName1 = input1->getName();
    //const auto inputName2 = input2->getName();
    //const auto inputDims = input1->getDimensions();
    //int32_t inputC = inputDims.d[1];
    //int32_t inputH = inputDims.d[2];
    //int32_t inputW = inputDims.d[3];

    //std::cout << inputName1 << " " << inputName2 << std::endl;
    //std::cout << inputC << " " << inputH << " " << inputW << std::endl;

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
    nvinfer1::ICudaEngine *engine_ = runtime_->deserializeCudaEngine(buffer.data(), size);
    if (!engine_) {
        throw std::runtime_error("Unable to deserialize cuda engine!");
    }
    context_ = engine_->createExecutionContext();
    assert(context_ != nullptr);
    std::cout << "Done" << std::endl;

    input1_index_ = engine_->getBindingIndex("input_1");
    input2_index_ = engine_->getBindingIndex("input_2");
    output_index_ = engine_->getBindingIndex("output");
    latent_index_ = engine_->getBindingIndex("latent");

    auto dims1 = engine_->getBindingDimensions(input1_index_);
    auto dims2 = engine_->getBindingDimensions(input2_index_);
    auto dims3 = engine_->getBindingDimensions(output_index_);
    auto dims4 = engine_->getBindingDimensions(latent_index_);

    input1_size_ = dims1.d[0] * dims1.d[1] * dims1.d[2] * dims1.d[3];
    input2_size_ = dims2.d[0] * dims2.d[1] * dims2.d[2] * dims2.d[3];
    output_size_ = dims3.d[0] * dims3.d[1] * dims3.d[2] * dims3.d[3];
    latent_size_ = dims4.d[0] * dims4.d[1] * dims4.d[2] * dims4.d[3];
}

EnlightenGAN::~EnlightenGAN() {
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
}

void EnlightenGAN::predict(cv::Mat image, cv::OutputArray enlighted) {
    cv::Mat imagef32(image.size(), CV_32FC3, cv::Scalar::all(0)), gray(image.size(), CV_32FC1, cv::Scalar(0));
    const uchar *row_input = image.data, *op_input;
    float *row_image = reinterpret_cast<float *>(imagef32.data), *row_gray = reinterpret_cast<float *>(gray.data), *op_image, vr, vg, vb;
    int ss = image.cols * image.rows, offset;
    for (int s = 0; s < ss; ++s) {
        offset = s * 3;
        op_input = row_input + offset;
        op_image = row_image + offset;

        vr = (static_cast<float>(op_input[2]) / 255.0 - 0.5) / 0.5;
        vg = (static_cast<float>(op_input[1]) / 255.0 - 0.5) / 0.5;
        vb = (static_cast<float>(op_input[0]) / 255.0 - 0.5) / 0.5;

        op_image[0] = vr;
        op_image[1] = vg;
        op_image[2] = vb;

        row_gray[s] = 1.0 - ((vr + 1) * 0.299 + (vg + 1) * 0.587 + (vb + 1) * 0.114) / 2.0;
    }

    const int height = imagef32.rows;
    const int width = imagef32.cols;
    cv::Mat blob_image = cv::dnn::blobFromImage(imagef32);
    void *buffers[4];
    float *outdata = new float[output_size_];


    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&buffers[input1_index_], input1_size_ * sizeof(float));
    cudaMalloc(&buffers[input2_index_], input2_size_ * sizeof(float));
    cudaMalloc(&buffers[output_index_], output_size_ * sizeof(float));
    cudaMalloc(&buffers[latent_index_], latent_size_ * sizeof(float));

    cudaMemcpyAsync(buffers[input1_index_], blob_image.data, input1_size_ * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(buffers[input2_index_], gray.data, input2_size_ * sizeof(float), cudaMemcpyHostToDevice, stream);
    context_->enqueue(1, buffers, stream, nullptr);
    cudaMemcpyAsync(outdata, buffers[output_index_], output_size_ * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    enlighted.create(height, width, CV_8UC3);
    cv::Mat _enlighted = enlighted.getMat(-1);
    int i = 0;
    for (int c = 2; c >= 0; --c) {
        for (int h = 0; h < height; ++h) {
            uchar *row_ptr = _enlighted.ptr<uchar>(h);
            for (int w = 0; w < width; ++w) {
                row_ptr[w * 3 + c] = cv::saturate_cast<uchar>(outdata[i++] * 127.5 + 127.5);
            }
        }
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[input1_index_]);
    cudaFree(buffers[input2_index_]);
    cudaFree(buffers[output_index_]);
    cudaFree(buffers[latent_index_]);
    
    delete[] outdata;
}

