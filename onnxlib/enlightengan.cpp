#include "EnlightenGAN.h"
#include "EnlightenGAN.h"
#include <iostream>
#include <vector>
//#include "cuda_provider_factory.h"
//#include "onnxruntime_session_options_config_keys.h"

EnlightenGAN::EnlightenGAN(const TCharString &model_path, bool use_gpu, int device_id) : ONNXAbsPredcitor(model_path, use_gpu, device_id) {

}

EnlightenGAN::~EnlightenGAN() {
}

void EnlightenGAN::predict(cv::Mat input, cv::OutputArray out) {
    cv::Mat imagef32(input.size(), CV_32FC3, cv::Scalar::all(0)), gray(input.size(), CV_32FC1, cv::Scalar(0));
    const uchar *row_input = input.data, *op_input;
    float *row_image = reinterpret_cast<float *>(imagef32.data), *row_gray = reinterpret_cast<float *>(gray.data), *op_image, vr, vg, vb;
    int ss = input.cols * input.rows, offset;
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
    
    std::vector<float> image_tensor_data, gray_tensor_data;

    Ort::Value image_tensor = mat2Tensor(imagef32, memory_info_, image_tensor_data);
    Ort::Value gray_tensor = mat2Tensor(gray, memory_info_, gray_tensor_data);

    //assert(image_tensor.IsTensor());
    //assert(gray_tensor.IsTensor());

    std::vector<Ort::Value> ort_inputs;
    ort_inputs.emplace_back(std::move(image_tensor));
    ort_inputs.emplace_back(std::move(gray_tensor));
    
    auto outputs = session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), output_names_.size());
    float *out_data = outputs.front().GetTensorMutableData<float>();

    out.create(height, width, CV_8UC3);

    cv::Mat _out = out.getMat(-1);
    int i = 0;
    for (int c = 2; c >= 0; --c) {
        for (int h = 0; h < height; ++h) {
            uchar *row_ptr = _out.ptr<uchar>(h);
            for (int w = 0; w < width; ++w) {
                row_ptr[w * 3 + c] = cv::saturate_cast<uchar>(out_data[i++] * 127.5 + 127.5);
            }
        }
    }
}
