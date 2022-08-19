#include "FBAMatting.h"

const int FBAMatting::L = 320;
const float FBAMatting::L1 = 2 * ((0.02 * L) * (0.02 * L));
const float FBAMatting::L2 = 2 * ((0.08 * L) * (0.08 * L));
const float FBAMatting::L3 = 2 * ((0.16 * L) * (0.16 * L));

FBAMatting::FBAMatting(const TCharString &model_path, bool use_gpu, int device_id) : ONNXAbsPredcitor(model_path, use_gpu, device_id) {
	
}

FBAMatting::~FBAMatting() {
}

void FBAMatting::predict(cv::Mat input, cv::OutputArray out) {
	cv::Mat imagef32(input.size(), CV_32FC3, cv::Scalar::all(0)), trimapf32(input.size(), CV_32FC1, cv::Scalar::all(0));
	cv::Mat trimapsf32, imageTransformed, trimapTransformed;
	uchar *row_input, *v4b_input, *data_input = input.data, *pixels_input;
	float *row_image, *v3f_image, *row_trimap, *pixels_image;
	float *data_image = reinterpret_cast<float *>(imagef32.data);
	float *data_trimap = reinterpret_cast<float *>(trimapf32.data);
	for (int r = 0; r < input.rows; ++r) {
		const int rcols = r * input.cols;
		row_input = data_input + (rcols * 4);
		row_image = data_image + (rcols * 3);
		row_trimap = data_trimap + rcols;
		for (int c = 0; c < input.cols; ++c) {
			pixels_input = row_input + c * 4;
			pixels_image = row_image + c * 3;
			row_trimap[c] = static_cast<float>(pixels_input[3]) / 255.0f;
			pixels_image[0] = static_cast<float>(pixels_input[2]) / 255.0f;
			pixels_image[1] = static_cast<float>(pixels_input[1]) / 255.0f;
			pixels_image[2] = static_cast<float>(pixels_input[0]) / 255.0f;
		}
	}

	imageTransformed = normalizeImage(imagef32);
	trimapsf32 = generatedTrimap(trimapf32);
	trimapTransformed = trimapTransform(trimapsf32);


	

	std::vector<float> image_tensor_data, image_transformed_tensor_data;
	std::vector<float> trimap_tensor_data, trimap_transformed_tensor_data;

	Ort::Value image_tensor = mat2Tensor(imagef32, memory_info_, image_tensor_data);
	Ort::Value trimap_tensor = mat2Tensor(trimapsf32, memory_info_, trimap_tensor_data);
	Ort::Value image_transformed_tensor = mat2Tensor(imagef32, memory_info_, image_transformed_tensor_data);
	Ort::Value trimap_transformed_tensor = mat2Tensor(trimapTransformed, memory_info_, trimap_transformed_tensor_data);

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.emplace_back(std::move(image_tensor));
	ort_inputs.emplace_back(std::move(trimap_tensor));
	ort_inputs.emplace_back(std::move(image_transformed_tensor));
	ort_inputs.emplace_back(std::move(trimap_transformed_tensor));

	auto outputs = session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), output_names_.size());
	float *out_tensor_data = outputs.front().GetTensorMutableData<float>();

	out.create(input.size(), CV_8UC4);
	cv::Mat _out = out.getMat(-1);

	std::vector<cv::Mat> trimapChannels;
	cv::split(trimapsf32, trimapChannels);

	const int height = imagef32.rows;
	const int width = imagef32.cols;
	size_t size_per_channel = height * width;
	const float *alpha_data = out_tensor_data;
	const float *fg_data = out_tensor_data + size_per_channel;
	for (int h = 0; h < height; ++h) {
		const float *pTrimap0 = trimapChannels.at(0).ptr<float>(h);
		const float *pTrimap1 = trimapChannels.at(1).ptr<float>(h);

		const float *pImage = imagef32.ptr<float>(h), *opImage;
		uchar *pOut = _out.ptr<uchar>(h), *opOut;

		for (int w = 0; w < width; ++w) {
			int pos = h * width + w;
			opOut = pOut + w * 4;
			// alpha post process
			if (std::fabs(pTrimap0[w] - 1) < 1e-3) {
				opOut[3] = 0;
			}
			else if (std::fabs(pTrimap1[w] - 1) < 1e-3) {
				opOut[3] = 255;
			}
			else {
				opOut[3] = cv::saturate_cast<uchar>(255 * alpha_data[pos]);
			}
			// fg post process
			if (std::fabs(alpha_data[pos] - 1) < 1e-3) {
				opImage = pImage + w * 3;
				opOut[0] = cv::saturate_cast<uchar>(opImage[2] * 255);
				opOut[1] = cv::saturate_cast<uchar>(opImage[1] * 255);
				opOut[2] = cv::saturate_cast<uchar>(opImage[0] * 255);
			}
			else {
				opOut[0] = cv::saturate_cast<uchar>(fg_data[size_per_channel * 2 + pos] * 255);
				opOut[1] = cv::saturate_cast<uchar>(fg_data[size_per_channel * 1 + pos] * 255);
				opOut[2] = cv::saturate_cast<uchar>(fg_data[pos] * 255);
			}
		}
	}
}

void FBAMatting::predict(const cv::Mat image, const cv::Mat trimap, cv::OutputArray fg, cv::OutputArray alpha) {
	cv::Mat imagef32, trimapf32, trimapsf32, imageTransformed, trimapTransformed;
	image.convertTo(imagef32, CV_32FC3, 1 / 255.0);
	trimap.convertTo(trimapf32, CV_32FC1, 1 / 255.0);
	cv::cvtColor(imagef32, imagef32, cv::COLOR_BGR2RGB);
	trimapsf32 = generatedTrimap(trimapf32);
	imageTransformed = normalizeImage(imagef32);
	trimapTransformed = trimapTransform(trimapsf32);

	const int height = imagef32.rows;
	const int width = imagef32.cols;

	std::vector<float> image_tensor_data, image_transformed_tensor_data;
	std::vector<float> trimap_tensor_data, trimap_transformed_tensor_data;

	Ort::Value image_tensor = mat2Tensor(imagef32, memory_info_, image_tensor_data);
	Ort::Value trimap_tensor = mat2Tensor(trimapsf32, memory_info_, trimap_tensor_data);
	Ort::Value image_transformed_tensor = mat2Tensor(imagef32, memory_info_, image_transformed_tensor_data);
	Ort::Value trimap_transformed_tensor = mat2Tensor(trimapTransformed, memory_info_, trimap_transformed_tensor_data);

	//assert(image_tensor.IsTensor());
	//assert(trimap_tensor.IsTensor());

	std::vector<Ort::Value> ort_inputs;
	ort_inputs.emplace_back(std::move(image_tensor));
	ort_inputs.emplace_back(std::move(trimap_tensor));
	ort_inputs.emplace_back(std::move(image_transformed_tensor));
	ort_inputs.emplace_back(std::move(trimap_transformed_tensor));

	auto outputs = session_.Run(Ort::RunOptions{ nullptr }, input_names_.data(), ort_inputs.data(), ort_inputs.size(), output_names_.data(), output_names_.size());
	float *out_tensor_data = outputs.front().GetTensorMutableData<float>();

	cv::Mat tmp_alpha(height, width, CV_32FC1, out_tensor_data);
	getFinalAlpha(tmp_alpha, trimapsf32, alpha);

	float *fg_data = out_tensor_data + height * width;
	cv::Mat tmpFg(height, width, CV_32FC3, cv::Scalar::all(0));

	int i = 0;
	for (int c = 2; c >= 0; --c) {
		for (int h = 0; h < height; ++h) {
			float *row_ptr = tmpFg.ptr<float>(h);
			for (int w = 0; w < width; ++w) {
				row_ptr[w * 3 + c] = fg_data[i++];
			}
		}
	}
	getFinalFg(imagef32, tmp_alpha, tmpFg, fg);
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
			// 绝对背景
			if (std::fabs(pTrimap0[c] - 1) < 1e-3) {
				pAlpha[c] = 0;
			}
			// 绝对前景
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
			// 绝对前景
			if (std::fabs(pAlpha[c] - 1) < 1e-3) {
				opFg = pFg + cc;
				opImage = pImage + cc;
				opFg[0] = cv::saturate_cast<uchar>(opImage[2] * 255);
				opFg[1] = cv::saturate_cast<uchar>(opImage[1] * 255);
				opFg[2] = cv::saturate_cast<uchar>(opImage[0] * 255);
			}
			// 
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
