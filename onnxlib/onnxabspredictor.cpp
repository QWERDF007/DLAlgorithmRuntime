#include "onnxabspredictor.h"

ONNXAbsPredcitor::ONNXAbsPredcitor(const TCharString &model_path, bool use_gpu, int device_id) :
	model_path_(model_path), env_(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default")),
	memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPUInput)), 
	use_gpu_(use_gpu), device_id_(device_id) {
	createSession();
	Ort::AllocatorWithDefaultOptions allocator;

	size_t num_input_nodes = session_.GetInputCount();
	size_t num_output_nodes = session_.GetOutputCount();
	std::vector<const char *> input_node_names(num_input_nodes);
	std::vector<const char *> output_node_names(num_output_nodes);

	for (int i = 0; i < num_input_nodes; ++i) {
		char *input_name = session_.GetInputName(i, allocator);
		input_names_.emplace_back(input_name);
	}

	for (int i = 0; i < num_output_nodes; ++i) {
		char *output_name = session_.GetOutputName(i, allocator);
		output_names_.emplace_back(output_name);
	}
}

ONNXAbsPredcitor::~ONNXAbsPredcitor() {
}

void ONNXAbsPredcitor::createSession() {
	Ort::SessionOptions session_options;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	if (use_gpu_) {
		//OrtCUDAProviderOptions options;
		//options.device_id = device_id_;
		//options.arena_extend_strategy = 0;
		//options.arena_extend_strategy = 0;
		//options.cuda_mem_limit = 4 * 1024 * 1024 * 1024;
		//options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
		//options.do_copy_in_default_stream = 1;
		//session_options.AppendExecutionProvider_CUDA(options);
		//Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id_));
	}
	session_ = Ort::Session(env_, model_path_.c_str(), session_options);
}

Ort::Value ONNXAbsPredcitor::mat2Tensor(cv::Mat image, const Ort::MemoryInfo &memory_info_handler, std::vector<float> &tensor_data_handler) {
	const int h = image.rows;
	const int w = image.cols;
	const int c = image.channels();
	const size_t elements_per_channel = h * w;
	const size_t elements = elements_per_channel * c;
	std::vector<int64_t> dims({ 1,c,h,w });
	//tensor_data_handler.resize(total);
	if (c > 1) {
		std::vector<cv::Mat> channels;
		cv::split(image, channels);
		for (cv::Mat &chn : channels) {
			float *pdata = reinterpret_cast<float *>(chn.data);
			tensor_data_handler.insert(tensor_data_handler.end(), pdata, pdata + elements_per_channel);
		}
	}
	else {
		float *pdata = reinterpret_cast<float *>(image.data);
		tensor_data_handler.insert(tensor_data_handler.end(), pdata, pdata + elements);
	}
	return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_data_handler.data(), elements, dims.data(), dims.size());
}
