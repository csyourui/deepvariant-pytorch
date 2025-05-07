cmake_minimum_required(VERSION 3.12)
project(onnx_model_test)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# 检查平台和架构
if(APPLE)
    # 检测处理器架构
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    message(STATUS "Detected architecture: ${ARCH}")
    
    # 为 Apple Silicon 设置特定优化
    if(ARCH STREQUAL "arm64")
        message(STATUS "Enabling Apple Silicon (M1/M2/M3) specific optimizations")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mcpu=apple-a14")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-a14")
    endif()
    
    # 启用 Metal 支持
    find_library(METAL_LIBRARY Metal REQUIRED)
    find_library(FOUNDATION_LIBRARY Foundation REQUIRED)
    find_library(MODELIO_LIBRARY ModelIO REQUIRED)
    find_library(METALKIT_LIBRARY MetalKit REQUIRED)
    find_library(ACCELERATE_LIBRARY Accelerate REQUIRED)
    
    message(STATUS "Found Metal: ${METAL_LIBRARY}")
    message(STATUS "Found Foundation: ${FOUNDATION_LIBRARY}")
    message(STATUS "Found ModelIO: ${MODELIO_LIBRARY}")
    message(STATUS "Found MetalKit: ${METALKIT_LIBRARY}")
    message(STATUS "Found Accelerate: ${ACCELERATE_LIBRARY}")
    
    # 定义使用 Metal
    add_definitions(-DUSE_METAL)
else()
    # 在非 Apple 平台上检测架构
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Detected architecture: ${ARCH}")
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 使用 FetchContent 下载并配置 ONNX Runtime
include(FetchContent)
set(ONNXRUNTIME_VERSION "1.21.0")

# 根据平台和架构选择正确的 ONNX Runtime 包
if(APPLE)
    if(ARCH STREQUAL "arm64")
        # macOS ARM64
        set(ONNXRUNTIME_PACKAGE "onnxruntime-osx-arm64-${ONNXRUNTIME_VERSION}.tgz")
        message(STATUS "Using macOS ARM64 package: ${ONNXRUNTIME_PACKAGE}")
    else()
        # macOS x86_64
        set(ONNXRUNTIME_PACKAGE "onnxruntime-osx-x86_64-${ONNXRUNTIME_VERSION}.tgz")
        message(STATUS "Using macOS x86_64 package: ${ONNXRUNTIME_PACKAGE}")
    endif()
else()
    # 检测是否为 Linux
    if(UNIX AND NOT APPLE)
        if(ARCH STREQUAL "aarch64")
            # Linux ARM64
            set(ONNXRUNTIME_PACKAGE "onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz")
            message(STATUS "Using Linux ARM64 package: ${ONNXRUNTIME_PACKAGE}")
        else()
            # Linux x86_64
            if(USE_GPU)
                set(ONNXRUNTIME_PACKAGE "onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz")
                message(STATUS "Using Linux x86_64 GPU package: ${ONNXRUNTIME_PACKAGE}")
            else()
                set(ONNXRUNTIME_PACKAGE "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz")
                message(STATUS "Using Linux x86_64 CPU package: ${ONNXRUNTIME_PACKAGE}")
            endif()
        endif()
    else()
        message(FATAL_ERROR "Unsupported platform. Only macOS and Linux are supported.")
    endif()
endif()

FetchContent_Declare(
    onnxruntime
    URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_PACKAGE}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# 配置 FetchContent 不要自动填充
FetchContent_GetProperties(onnxruntime)
if(NOT onnxruntime_POPULATED)
    message(STATUS "Fetching ONNX Runtime ${ONNXRUNTIME_VERSION} (${ONNXRUNTIME_PACKAGE})")
    FetchContent_Populate(onnxruntime)
endif()

# 设置 ONNXRUNTIME 包含路径和库路径
set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${onnxruntime_SOURCE_DIR}/lib")

# 根据平台选择正确的库文件扩展名
if(APPLE)
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.${ONNXRUNTIME_VERSION}.dylib")
else()
    set(ONNXRUNTIME_LIB "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.${ONNXRUNTIME_VERSION}")
endif()

# 检查库文件是否存在
if(NOT EXISTS ${ONNXRUNTIME_LIB})
    message(FATAL_ERROR "ONNX Runtime library not found at ${ONNXRUNTIME_LIB}")
endif()

# 添加 ONNX Runtime 包含目录
include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# 添加测试程序
add_executable(test_inception3 test/test_inception3.cpp)

# 链接ONNX Runtime库
target_link_libraries(test_inception3 ${ONNXRUNTIME_LIB})

# 添加平台特定选项
if(UNIX AND NOT APPLE)
    target_link_libraries(test_inception3 pthread dl)
endif()

# 可选：添加CUDA支持
option(WITH_CUDA "Build with CUDA support" OFF)
if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    add_definitions(-DUSE_CUDA)
    target_include_directories(test_inception3 PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(test_inception3 ${CUDA_LIBRARIES})
endif()

# 输出配置信息
message(STATUS "ONNX Runtime Include Dir: ${ONNXRUNTIME_DIR}/include")
message(STATUS "ONNX Runtime Library: ${ONNXRUNTIME_LIB}")
