#include "FBAMatting.h"

const int FBAMatting::L = 320;
const float FBAMatting::L1 = 2 * ((0.02 * L) * (0.02 * L));
const float FBAMatting::L2 = 2 * ((0.08 * L) * (0.08 * L));
const float FBAMatting::L3 = 2 * ((0.16 * L) * (0.16 * L));

FBAMatting::FBAMatting(std::string model, int device_id) : _device(torch::Device(torch::kCUDA, device_id)) {
	std::cout << "Loading model from " << model << std::endl;
	_module = torch::jit::load(model.c_str(), _device);
	std::cout << "done!"  << std::endl;
	//_module.to(_device);
	_module.eval();
	setThreads(6);
}

FBAMatting::~FBAMatting() {
}

void FBAMatting::predict(const cv::Mat &image, const cv::Mat &trimap, cv::OutputArray fg, cv::OutputArray alpha) {
	torch::NoGradGuard no_grad;
	int height = image.rows, width = image.cols;
	torch::Tensor imageTensor, trimapTensor, imageTransformedTensor, trimapTransformedTensor;
	cv::Mat imagef32, trimapf32, trimapsf32, imageTransformed, trimapTransformed;
	image.convertTo(imagef32, CV_32FC3, 1 / 255.0);
	trimap.convertTo(trimapf32, CV_32FC1, 1 / 255.0);
	cv::cvtColor(imagef32, imagef32, cv::COLOR_BGR2RGB);
	trimapsf32 = generatedTrimap(trimapf32);
	imageTransformed = normalizeImage(imagef32);
	trimapTransformed = trimapTransform(trimapsf32);

	mat2Tensor(imagef32, &imageTensor);
	mat2Tensor(trimapsf32, &trimapTensor);
	mat2Tensor(imageTransformed, &imageTransformedTensor);
	mat2Tensor(trimapTransformed, &trimapTransformedTensor);
	
	std::vector<torch::jit::IValue> inputs;
	//inputs.push_back(imageTensor.to(torch::kCPU));
	//inputs.push_back(trimapTensor.to(torch::kCPU));
	//inputs.push_back(imageTransformedTensor.to(torch::kCPU));
	//inputs.push_back(trimapTransformedTensor.to(torch::kCPU));
	inputs.push_back(imageTensor.to(_device));
	inputs.push_back(trimapTensor.to(_device));
	inputs.push_back(imageTransformedTensor.to(_device));
	inputs.push_back(trimapTransformedTensor.to(_device));
	auto outTensor = _module.forward(inputs).toTensor();

	auto alphaTensor = outTensor.slice(1, 0, 1, 1);
	auto fgTensor = outTensor.slice(1, 1, 4, 1);
	cv::Mat tmpAlpha(height, width, CV_32FC1, cv::Scalar::all(0));
	tensor2Mat(alphaTensor, &tmpAlpha, 0);
	getFinalAlpha(tmpAlpha, trimapsf32, alpha);
	cv::Mat tmpFg(height, width, CV_32FC3, cv::Scalar::all(0));
	tensor2Mat(fgTensor, &tmpFg);
	getFinalFg(imagef32, tmpAlpha, tmpFg, fg);
}

void FBAMatting::setThreads(int threads) {
	torch::init_num_threads();
	torch::set_num_threads(threads);
}

int FBAMatting::threads() {
	return torch::get_num_threads();
}

void FBAMatting::padTransform(cv::Mat image, cv::Mat trimap, cv::OutputArray outImage, cv::OutputArray outTrimap) {
	int modcols = image.cols % 8, modrows = image.rows % 8;
	int recols = modcols == 0 ? image.cols : (image.cols + 8 - modcols);
	int rerows = modrows == 0 ? image.rows : (image.rows + 8 - modrows);
	if (modcols == 0 && modrows == 0) {
		image.copyTo(outImage);
		trimap.copyTo(outTrimap);
	} else {
		cv::Mat tmpImage(rerows, recols, image.type(), cv::Scalar::all(0));
		cv::Mat tmpTrimap(rerows, recols, trimap.type(), cv::Scalar::all(0));
		cv::Rect r(0, 0, image.cols, image.rows);
		image.copyTo(tmpImage(r));
		trimap.copyTo(tmpTrimap(r));
		tmpImage.copyTo(outImage);
		tmpTrimap.copyTo(outTrimap);
	}
}

cv::Mat FBAMatting::trimapTransform(const cv::Mat &trimapF32) {
	int height = trimapF32.rows, width = trimapF32.cols;
	std::vector<cv::Mat> clicks;
	for (int i = 0; i < 6; ++i) {
		clicks.push_back(cv::Mat(trimapF32.size(), CV_32FC1, cv::Scalar(0)));
	}
	std::vector<cv::Mat> trimapChannels;
	cv::split(trimapF32, trimapChannels);

#pragma omp parallel num_threads(2)
	{
#pragma omp sections
		{
#pragma omp section
			{
				cv::Mat trimapChannel = trimapChannels.at(0);
				if (cv::countNonZero(trimapChannel) > 0) {
					cv::Mat tmp = 1 - trimapChannel;
					cv::Mat a;
					tmp.convertTo(a, CV_8UC1, 255);
					cv::Mat dt;
					cv::Mat tmp2(a.rows, a.cols, CV_32FC1, cv::Scalar(0));
					cv::distanceTransform(a, dt, cv::DIST_L2, CV_32F);
					cv::Mat dt_mask = tmp2 - dt.mul(dt);
					cv::exp(dt_mask / L1, clicks.at(0 * 3));
					cv::exp(dt_mask / L2, clicks.at(0 * 3 + 1));
					cv::exp(dt_mask / L3, clicks.at(0 * 3 + 2));
				}
			}
#pragma omp section
			{
				cv::Mat trimapChannel = trimapChannels.at(1);
				if (cv::countNonZero(trimapChannel) > 0) {
					cv::Mat tmp = 1 - trimapChannel;
					cv::Mat a;
					tmp.convertTo(a, CV_8UC1, 255);
					cv::Mat dt;
					cv::Mat tmp2(a.rows, a.cols, CV_32FC1, cv::Scalar(0));
					cv::distanceTransform(a, dt, cv::DIST_L2, CV_32F);
					cv::Mat dt_mask = tmp2 - dt.mul(dt);
					cv::exp(dt_mask / L1, clicks.at(1 * 3));
					cv::exp(dt_mask / L2, clicks.at(1 * 3 + 1));
					cv::exp(dt_mask / L3, clicks.at(1 * 3 + 2));
				}
			}
		}
	}
	cv::Mat click;
	cv::merge(clicks, click);
	return click;
}

cv::Mat FBAMatting::normalizeImage(const cv::Mat &imageF32) {
	int height = imageF32.rows, width = imageF32.cols;
	cv::Mat imageMean(height, width, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
	cv::Mat imageStd(height, width, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
	cv::Mat imagetransformed(height, width, CV_32FC3, cv::Scalar::all(0));
	imagetransformed = (imageF32 - imageMean) / imageStd;
	return imagetransformed;
}

cv::Mat FBAMatting::generatedTrimap(const cv::Mat &trimapF32) {
	int height = trimapF32.rows, width = trimapF32.cols;
	cv::Mat resTrimap;
	cv::Mat trimap0(trimapF32.rows, trimapF32.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat trimap1(trimapF32.rows, trimapF32.cols, CV_32FC1, cv::Scalar(0));
	std::vector<cv::Mat> trimaps;
	for (int r = 0; r < height; ++r) {
		const float *pTrimap = trimapF32.ptr<float>(r);
		float *pTrimap0 = trimap0.ptr<float>(r);
		float *pTrimap1 = trimap1.ptr<float>(r);
		for (int c = 0; c < width; ++c) {
			if (std::fabs(pTrimap[c] - 1) < 1e-3)
				pTrimap1[c] = 1;
			if (std::fabs(pTrimap[c] - 0) < 1e-3)
				pTrimap0[c] = 1;
		}
	}
	trimaps.push_back(trimap0);
	trimaps.push_back(trimap1);
	cv::merge(trimaps, resTrimap);
	return resTrimap;
}

void FBAMatting::getFinalAlpha(const cv::Mat &predAlpha, const cv::Mat &trimaps, cv::OutputArray alpha) {
	int height = predAlpha.rows, width = predAlpha.cols;
	if (alpha.empty()) {
		alpha.create(predAlpha.size(), CV_8UC1);
	}
	cv::Mat _alpha = alpha.getMat(-1);
	std::vector<cv::Mat> trimapChannels;
	cv::split(trimaps, trimapChannels);
	for (int r = 0; r < height; ++r) {
		const float *pPred = predAlpha.ptr<float>(r);
		const float *pTrimap0 = trimapChannels.at(0).ptr<float>(r);
		const float *pTrimap1 = trimapChannels.at(1).ptr<float>(r);
		uchar *pAlpha = _alpha.ptr<uchar>(r);
		for (int c = 0; c < width; ++c) {
			if (std::fabs(pTrimap0[c] - 1) < 1e-3) {
				pAlpha[c] = 0;
			}
			else if (std::fabs(pTrimap1[c] - 1) < 1e-3) {
				pAlpha[c] = 255;
			}
			else {
				pAlpha[c] = cv::saturate_cast<uchar>(255 * pPred[c]);
			}
		}
	}
}

void FBAMatting::getFinalFg(const cv::Mat &image, const cv::Mat &predAlpha, const cv::Mat &predFg, cv::OutputArray fg) {
	int height = image.rows, width = image.cols;
	cv::Mat tmpFg(image.size(), CV_8UC3, cv::Scalar::all(0));
	for (int r = 0; r < height; ++r) {
		const float *pImage = image.ptr<float>(r), *opImage;
		const float *pTmpFg = predFg.ptr<float>(r), *opTmpFg;
		const float *pAlpha = predAlpha.ptr<float>(r);
		
		uchar *pFg = tmpFg.ptr<uchar>(r), *opFg;
		for (int c = 0; c < width; ++c) {
			int cc = c * 3;
			if (std::fabs(pAlpha[c] - 1) < 1e-3) {
				opFg = pFg + cc;
				opImage = pImage + cc;
				opFg[0] = cv::saturate_cast<uchar>(opImage[2] * 255);
				opFg[1] = cv::saturate_cast<uchar>(opImage[1] * 255);
				opFg[2] = cv::saturate_cast<uchar>(opImage[0] * 255);
			}
			else {
				opFg = pFg + cc;
				opTmpFg = pTmpFg + cc;
				opFg[0] = cv::saturate_cast<uchar>(opTmpFg[2] * 255);
				opFg[1] = cv::saturate_cast<uchar>(opTmpFg[1] * 255);
				opFg[2] = cv::saturate_cast<uchar>(opTmpFg[0] * 255);
			}
		}
	}
	tmpFg.copyTo(fg);
}

void FBAMatting::mat2Tensor(const cv::Mat &imageFloat, torch::Tensor *outTensor) {
	int height = imageFloat.rows;
	int width = imageFloat.cols;
	int channel = imageFloat.channels();
	auto tensor = torch::from_blob(imageFloat.data, { 1, height, width, channel });
	*outTensor = tensor.permute({ 0, 3, 1, 2 });
}

void FBAMatting::tensor2Mat(const torch::Tensor &inputTensor, cv::Mat *imageFloat, int type) {
	std::vector<cv::Mat> images;
	torch::Tensor tensor = inputTensor.squeeze(0).detach().to(torch::kCPU);
	int channel = tensor.size(0);
	int height = tensor.size(1);
	int width = tensor.size(2);
	for (int i = 0; i < channel; ++i) {
		cv::Mat image(height, width, CV_32FC1, tensor[i].data_ptr<float>());
		images.push_back(image);

	}
	cv::merge(images, *imageFloat);
}
