#include "DeepLabv3Plus.h"

DeepLabv3Plus::DeepLabv3Plus(std::string &model_path, int device_id) : device_(torch::Device(torch::kCUDA, device_id)) {
	module_ = torch::jit::load(model_path.c_str(), device_);
	module_.eval();
	setThreads(4);
}

DeepLabv3Plus::~DeepLabv3Plus() {

}

void DeepLabv3Plus::predict(cv::Mat image, cv::OutputArray segment) {
	cv::Mat rgb;
	cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
	rgb.convertTo(rgb, CV_32FC3, 1, 0);
	rgb = (rgb - cv::Scalar(123.675, 116.28, 103.53)) / cv::Scalar(58.395, 57.12, 57.375);

	torch::NoGradGuard no_grad;
	torch::Tensor image_tensor;
	mat2Tensor(rgb, &image_tensor);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(image_tensor.to(device_));
	auto out_tensor = module_.forward(inputs).toTensor();
	auto t = torch::max(out_tensor[0], 0);
	auto maskTensor = std::get<1>(t).to(torch::kUInt8);
	cv::Mat tmp(image.size(), CV_8UC1);
	tensor2Mat(maskTensor, &tmp);
	tmp.copyTo(segment);
}

void DeepLabv3Plus::setThreads(int threads) {
	torch::init_num_threads();
	torch::set_num_threads(threads);
}

int DeepLabv3Plus::threads() {
	return torch::get_thread_num();
}

void DeepLabv3Plus::mat2Tensor(const cv::Mat &imageFloat, torch::Tensor *outTensor) {
	int height = imageFloat.rows;
	int width = imageFloat.cols;
	int channel = imageFloat.channels();
	auto tensor = torch::from_blob(imageFloat.data, { 1, height, width, channel });
	*outTensor = tensor.permute({ 0, 3, 1, 2 });
}

void DeepLabv3Plus::tensor2Mat(const torch::Tensor &inputTensor, cv::Mat *mask) {
	torch::Tensor tensor;
	tensor = inputTensor.detach().to(torch::kCPU);
	std::memcpy((void *)mask->data, tensor.data_ptr(), torch::elementSize(torch::kUInt8) * tensor.numel());
}
