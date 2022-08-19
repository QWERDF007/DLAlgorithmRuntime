#pragma once

#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>
#include <opencv2/opencv.hpp>


class __declspec(dllexport) FBAMatting {
public:
	FBAMatting(std::string model, int device_id = 0);
	~FBAMatting();

	void predict(const cv::Mat &image, const cv::Mat &trimap, cv::OutputArray fg, cv::OutputArray alpha);

	void setThreads(int threads);

	int threads();

private:

	torch::jit::script::Module _module;
	torch::Device _device;

	const static int L;
	const static float L1;
	const static float L2;
	const static float L3;

	void padTransform(cv::Mat image, cv::Mat trimap, cv::OutputArray outImage, cv::OutputArray outTrimap);

	cv::Mat trimapTransform(const cv::Mat &trimapF32);
	cv::Mat normalizeImage(const cv::Mat &imageF32);
	cv::Mat generatedTrimap(const cv::Mat &trimapF32);
	void getFinalAlpha(const cv::Mat &pred, const cv::Mat &trimaps, cv::OutputArray alpha);
	void getFinalFg(const cv::Mat &image, const cv::Mat &tmpAlpha, const cv::Mat &tmpFg, cv::OutputArray fg);

	void mat2Tensor(const cv::Mat &imageFloat, torch::Tensor *outTensor);
	void tensor2Mat(const torch::Tensor &inputTensor, cv::Mat *imageFloat, int type = 1);
};