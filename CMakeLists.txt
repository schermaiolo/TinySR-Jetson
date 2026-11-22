cmake_minimum_required(VERSION 3.20)
project(TinySRJetson VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY NAMES nvinfer REQUIRED)
if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT C++ headers not found. Install libnvinfer-dev.")
endif()

add_library(tinysr_runtime STATIC
    src/trt_engine.cpp
    src/image_processor.cpp
    src/pipeline.cpp
    src/cuda_kernels.cu)
target_compile_features(tinysr_runtime PUBLIC cxx_std_17)
set_target_properties(tinysr_runtime PROPERTIES CUDA_ARCHITECTURES 87)
target_include_directories(tinysr_runtime PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tinysr_runtime PUBLIC
    CUDA::cudart ${TENSORRT_LIBRARY} ${OpenCV_LIBS})

add_executable(tinysr-jetson src/main.cpp)
target_link_libraries(tinysr-jetson PRIVATE tinysr_runtime)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_image_processor
        tests/test_image_processor.cpp src/image_processor.cpp)
    target_include_directories(test_image_processor PRIVATE
        ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(test_image_processor PRIVATE ${OpenCV_LIBS})
    add_test(NAME image_processor COMMAND test_image_processor)
    add_test(NAME help COMMAND tinysr-jetson --help)
endif()
