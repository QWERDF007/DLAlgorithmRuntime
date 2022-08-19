cmake_minimum_required(VERSION 3.0)

set(PROJECT_NAME dldemo)

project(${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory("FBAMatting")
add_subdirectory("EnlightenGAN")
add_subdirectory("DeepLabv3Plus")

include_directories(
    $ENV{DEV_LIB_DIR}/cxxopts/include
    $ENV{DEV_LIB_DIR}/spdlog/include
    $ENV{DEV_LIB_DIR}/opencv/include
    $ENV{DEV_LIB_DIR}/libtorch/include
    $ENV{DEV_LIB_DIR}/libtorch/include/torch/csrc/api/include
)

link_directories(
    $ENV{DEV_LIB_DIR}/spdlog/lib
    $ENV{DEV_LIB_DIR}/opencv/lib
    $ENV{DEV_LIB_DIR}/libtorch/lib
)

link_libraries(
    spdlog
    c10 c10_cuda torch torch_cuda torch_cpu "-Wl,--no-as-needed -ltorch_cuda"
    opencv_world347
    FBAMatting
    EnlightenGAN
    DeepLabv3Plus
)

add_executable(${PROJECT_NAME} main.cpp)