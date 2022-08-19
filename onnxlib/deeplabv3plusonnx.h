#pragma once

#include <opencv2/opencv.hpp>
#include "onnxabspredictor.h"


class CRED_ONNX_API DeepLabv3PlusONNX : public ONNXAbsPredcitor {
public:
	DeepLabv3PlusONNX(const TCharString &model_path, bool use_gpu = false, int device_id = -1);

	~DeepLabv3PlusONNX();

	void predict(cv::Mat input, cv::OutputArray out) override;

private:

};