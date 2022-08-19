#pragma once

#include <opencv2/opencv.hpp>
#include "onnxabspredictor.h"


class CRED_ONNX_API EnlightenGAN : public ONNXAbsPredcitor {
public:
    EnlightenGAN(const TCharString &model_path, bool use_gpu = false, int device_id = -1);

    ~EnlightenGAN();

    void predict(cv::Mat input, cv::OutputArray output) override;

private:

};