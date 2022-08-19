#pragma once

#include <opencv2/opencv.hpp>
#include "onnxglobal.h"

class CRED_ONNX_API ONNXAbsPredcitor {
public:
    ONNXAbsPredcitor(const TCharString &model_path, bool use_gpu = false, int device_id = -1);

    ~ONNXAbsPredcitor();

    virtual void predict(cv::Mat input, cv::OutputArray out) = 0;

private:

    void createSession();

protected:
	bool use_gpu_ = false;
	int device_id_ = -1;
	std::vector<const char *> input_names_;
	std::vector<const char *> output_names_;
	Ort::Env env_;
	Ort::MemoryInfo memory_info_;
	const TCharString model_path_;
	Ort::Session session_{ nullptr };

	Ort::Value mat2Tensor(cv::Mat input, const Ort::MemoryInfo &memory_info_handler, std::vector<float> &tensor_data_handler);
};