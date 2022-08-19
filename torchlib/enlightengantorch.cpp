#include "EnlightenGAN.h"

EnlightenGAN::EnlightenGAN(std::string model, int device_id) : _device(torch::Device(torch::kCUDA, device_id)) {
	_module = torch::jit::load(model.c_str(), _device);
	_module.eval();
	setThreads(6);
}

EnlightenGAN::~EnlightenGAN() {
}

void EnlightenGAN::predict(cv::Mat image, cv::OutputArray enlighted) {
	cv::Mat _image(image.size(), CV_32FC3, cv::Scalar::all(0)), gray(image.size(), CV_32FC1, cv::Scalar(0));
	const uchar *row_input = image.data, *op_input;
	float *row_image = reinterpret_cast<float *>(_image.data), *row_gray = reinterpret_cast<float *>(gray.data), *op_image, vr, vg, vb;
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

	torch::NoGradGuard no_grad;
	torch::Tensor imageTensor, grayTensor;
	mat2Tensor(_image, &imageTensor);
	mat2Tensor(gray, &grayTensor);

	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(imageTensor.to(_device));
	inputs.push_back(grayTensor.to(_device));
	auto outTensor = _module.forward(inputs).toTuple();
	auto t = outTensor->elements()[0].toTensor();
	cv::Mat tmp(_image.size(), CV_32FC3, cv::Scalar::all(0));
	tensor2Mat(t, &tmp);
	enlighted.create(image.size(), CV_8UC3);
	cv::Mat _fakeImage = enlighted.getMat(-1);
	uchar *row_output = _fakeImage.data, *op_output;
	row_image = reinterpret_cast<float *>(tmp.data);
	for (int s = 0; s < ss; ++s) {
		offset = s * 3;
		op_output = row_output + offset;
		op_image = row_image + offset;
		op_output[2] = cv::saturate_cast<uchar>((op_image[0] / 2 + 0.5) * 255.0);
		op_output[1] = cv::saturate_cast<uchar>((op_image[1] / 2 + 0.5) * 255.0);
		op_output[0] = cv::saturate_cast<uchar>((op_image[2] / 2 + 0.5) * 255.0);
	}
	
}

void EnlightenGAN::setThreads(int threads) {
    torch::init_num_threads();
    torch::set_num_threads(threads);
}

int EnlightenGAN::threads() {
    return torch::get_thread_num(); 
}

void EnlightenGAN::mat2Tensor(const cv::Mat &imageFloat, torch::Tensor *outTensor) {
	int height = imageFloat.rows;
	int width = imageFloat.cols;
	int channel = imageFloat.channels();
	auto tensor = torch::from_blob(imageFloat.data, { 1, height, width, channel });
	*outTensor = tensor.permute({ 0, 3, 1, 2 });
}

void EnlightenGAN::tensor2Mat(const torch::Tensor &inputTensor, cv::Mat *imageFloat) {
	torch::Tensor tensor;
	tensor = inputTensor.squeeze().detach().permute({ 1, 2, 0 }).to(torch::kCPU);
	std::memcpy((void *)imageFloat->data, tensor.data_ptr(), torch::elementSize(torch::kF32) * tensor.numel());
	//std::vector<cv::Mat> images;
	//torch::Tensor tensor = inputTensor.squeeze(0).detach().to(torch::kCPU);
	//int channel = tensor.size(0);
	//int height = tensor.size(1);
	//int width = tensor.size(2);
	//for (int i = 0; i < channel; ++i) {
	//	cv::Mat image(height, width, CV_32FC1, tensor[i].data_ptr<float>());
	//	images.push_back(image);

	//}
	//cv::merge(images, *imageFloat);
}
