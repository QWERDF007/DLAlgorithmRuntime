#pragma once

#include <string>
#include <onnxruntime_cxx_api.h>
//#include <cuda_provider_factory.h>

using TCharString = std::basic_string<ORTCHAR_T>;

#ifdef _WIN32
#define CRED_ONNX_API __declspec(dllexport)
#else
#define CRED_ONNX_API
#endif
