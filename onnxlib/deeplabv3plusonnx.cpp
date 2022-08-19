#include "DeepLabv3PlusONNX.h"

DeepLabv3PlusONNX::DeepLabv3PlusONNX(const TCharString &model_path, bool use_gpu, int device_id) : ONNXAbsPredcitor(model_path, use_gpu, device_id) {

}

DeepLabv3PlusONNX::~DeepLabv3PlusONNX() {
}

void DeepLabv3PlusONNX::predict(cv::Mat input, cv::OutputArray out) {
	cv::Mat rgb;
	cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
	rgb.convertTo(rgb, CV_32FC3, 1, 0);
	rgb = (rgb - cv::Scalar(123.675, 116.28, 103.53)) / cv::Scalar(58.395, 57.12, 57.375);

	const int height = rgb.rows;
	const int width = rgb.cols;

	std::vector<float> image_tensor_data;
	Ort::Value image_tensor = mat2Tensor(rgb, memory_info_, image_tensor_data);

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.emplace_back(std::move(image_tensor));

	auto outputs = session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), output_names_.size());
	int64_t *out_data = outputs.front().GetTensorMutableData<int64_t>();

	out.create(height, width, CV_8UC1);
	cv::Mat _out = out.getMat(-1);

	size_t len = height * width;
	uchar *pdata = _out.data;
	for (int i = 0; i < len; ++i) {
		pdata[i] = static_cast<uchar>(out_data[i]);
	}
}

