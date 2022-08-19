#pragma once

#include <opencv2/opencv.hpp>
#include "onnxabspredictor.h"


class CRED_ONNX_API FBAMatting : public ONNXAbsPredcitor {
public:
	FBAMatting(const TCharString &model_path, bool use_gpu = false, int device_id = -1);

	~FBAMatting();

	void predict(cv::Mat input, cv::OutputArray out) override;

	void predict(const cv::Mat image, const cv::Mat trimap, cv::OutputArray fg, cv::OutputArray alpha);

private:

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
};