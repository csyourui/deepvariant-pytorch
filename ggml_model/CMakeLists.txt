cmake_minimum_required(VERSION 3.14)
project(inception3 VERSION 1.0.0 LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置构建类型
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 确保在 Release 模式下使用优化
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(APPLE)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    endif()
endif()

# 检查是否为 Apple 平台
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
endif()

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找线程库
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# 查找并启用 OpenMP (如果可用)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 引入 ggml 库
# 两种方式: 1. 自定义 ggml 路径或 2. 系统安装的 ggml
option(USE_SYSTEM_GGML "Use system installed ggml" OFF)

if(USE_SYSTEM_GGML)
    # 使用系统安装的 ggml
    find_package(ggml REQUIRED)
    
    # 如果找到系统 ggml，设置包含目录和库
    include_directories(${GGML_INCLUDE_DIR})
    set(GGML_LIBRARIES ggml)
else()
    # 使用自定义 ggml 库
    set(GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml" CACHE PATH "Path to ggml source")
    
    # 检查 ggml 源目录是否存在
    if(NOT EXISTS ${GGML_SOURCE_DIR})
        # message(FATAL_ERROR "GGML source directory not found at ${GGML_SOURCE_DIR}. 
        #     Please clone the ggml repository: 
        #     git clone https://github.com/ggerganov/ggml.git ${GGML_SOURCE_DIR}")
        include(FetchContent)
        FetchContent_Declare(
            ggml
            GIT_REPOSITORY https://github.com/ggml-org/ggml
        )
        FetchContent_MakeAvailable(ggml)
        # FetchContent 会创建一个变量 ggml_SOURCE_DIR，包含下载的源代码路径
        set(GGML_SOURCE_DIR ${ggml_SOURCE_DIR} CACHE PATH "Path to ggml source" FORCE)
        message(STATUS "GGML source directory not found. Cloned ggml repository.")
    endif()
    
    # 设置 GGML 编译选项 - 禁用测试和示例
    set(GGML_BUILD_TESTS OFF CACHE BOOL "Disable GGML tests" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Disable GGML examples" FORCE)
 
    include_directories(${GGML_SOURCE_DIR}/include)
    include_directories(${GGML_SOURCE_DIR}/src)
    add_subdirectory(${GGML_SOURCE_DIR} ${CMAKE_BINARY_DIR}/ggml-build)

    # 如果在 Apple 平台，添加 Metal 支持
    if(APPLE)
        list(APPEND GGML_SOURCES ${GGML_SOURCE_DIR}/src/ggml-metal/ggml-metal.m)
        # 设置 Metal 着色器文件
        set(METAL_SHADER ${GGML_SOURCE_DIR}/src/ggml-metal/ggml-metal.metal)
        include_directories(${GGML_SOURCE_DIR}/src/ggml-metal)
    endif()
    
endif()

############################################################################## 

# 添加pybind11支持
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(PYTHON_VERSION "Python version to use" "3.10") # 新增选项

if(BUILD_PYTHON_BINDINGS)
    # 设置首选 Python 版本
    set(Python_ADDITIONAL_VERSIONS "${PYTHON_VERSION}")

    # 更具体的 Python 查找
    find_package(Python ${PYTHON_VERSION} EXACT COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Found Python: ${Python_EXECUTABLE} (version: ${Python_VERSION})")
    message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python libraries: ${Python_LIBRARIES}")

    # 查找pybind11
    find_package(pybind11 CONFIG)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
    
    # 添加Python绑定构建
    include(${CMAKE_CURRENT_SOURCE_DIR}/pybind11_inception3_bindings.cmake)
endif()

############################################################################## 

# 设置 inception3 源文件
set(INECPTION3_SOURCES
    inception3.cpp
    utils.cpp
    test/test_inception3.cpp
)

# 添加自定义的头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# 添加可执行文件
add_executable(test_inception3 ${INECPTION3_SOURCES})

# 如果使用自定义 ggml，添加源文件到目标
if(NOT USE_SYSTEM_GGML)
    target_sources(test_inception3 PRIVATE ${GGML_SOURCES})
    
    # 为 Apple 平台添加 Metal 相关内容
    if(APPLE)
        target_sources(test_inception3 PRIVATE ${CMAKE_BINARY_DIR}/ggml-metal.metallib)
        set_source_files_properties(${CMAKE_BINARY_DIR}/ggml-metal.metallib PROPERTIES
            GENERATED TRUE
            MACOSX_PACKAGE_LOCATION Resources
        )
        
        # 告诉编译器我们正在使用 Metal
        target_compile_definitions(test_inception3 PRIVATE -DGGML_USE_METAL=1)
    endif()
    target_link_libraries(test_inception3 PRIVATE ggml ggml-base ggml-metal ggml-cpu ggml-blas)
else()
    target_link_libraries(test_inception3 PRIVATE ${GGML_LIBRARIES})
endif()

# 链接必要的库
target_link_libraries(test_inception3 PRIVATE Threads::Threads)

# 为 Apple 平台添加特定的库链接
if(APPLE)
    target_link_libraries(test_inception3 PRIVATE
        ${ACCELERATE_LIBRARY}
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${MODELIO_LIBRARY}
        ${METALKIT_LIBRARY}
    )
endif()

# 如果启用了 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_inception3 PRIVATE OpenMP::OpenMP_CXX)
endif()

# 添加更多编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_inception3 PRIVATE -Wall -Wextra)
endif()

############################################################################## 

# 设置 inception3 源文件
set(TEST_INECPTION3_SOURCES
    inception3.cpp
    utils.cpp
    test/test_inception3_layer.cpp
)

# 添加可执行文件，用于测试
add_executable(test_inception3_layer ${TEST_INECPTION3_SOURCES})

# 如果使用自定义 ggml，添加源文件到目标
if(NOT USE_SYSTEM_GGML)
    target_sources(test_inception3_layer PRIVATE ${GGML_SOURCES})
    
    # 为 Apple 平台添加 Metal 相关内容
    if(APPLE)
        target_sources(test_inception3_layer PRIVATE ${CMAKE_BINARY_DIR}/ggml-metal.metallib)
        set_source_files_properties(${CMAKE_BINARY_DIR}/ggml-metal.metallib PROPERTIES
            GENERATED TRUE
            MACOSX_PACKAGE_LOCATION Resources
        )
        
        # 告诉编译器我们正在使用 Metal
        target_compile_definitions(test_inception3_layer PRIVATE -DGGML_USE_METAL=1)
    endif()
    target_link_libraries(test_inception3_layer PRIVATE ggml ggml-base ggml-metal ggml-cpu ggml-blas)
else()
    target_link_libraries(test_inception3_layer PRIVATE ${GGML_LIBRARIES})
endif()

# 链接必要的库
target_link_libraries(test_inception3_layer PRIVATE Threads::Threads)

# 为 Apple 平台添加特定的库链接
if(APPLE)
    target_link_libraries(test_inception3_layer PRIVATE
        ${ACCELERATE_LIBRARY}
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${MODELIO_LIBRARY}
        ${METALKIT_LIBRARY}
    )
endif()

# 如果启用了 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_inception3_layer PRIVATE OpenMP::OpenMP_CXX)
endif()

# 添加更多编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_inception3_layer PRIVATE -Wall -Wextra)
endif()
##############################################################################


# 输出配置信息
message(STATUS "Configuration:")
message(STATUS "  CMake version       : ${CMAKE_VERSION}")
message(STATUS "  Generator           : ${CMAKE_GENERATOR}")
message(STATUS "  Build type          : ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler          : ${CMAKE_C_COMPILER}")
message(STATUS "  C++ compiler        : ${CMAKE_CXX_COMPILER}")
if(APPLE)
    message(STATUS "  Using Metal        : YES")
else()
    message(STATUS "  Using Metal        : NO")
endif()

# 显示所有变量（调试用）
# get_cmake_property(_variableNames VARIABLES)
# list(SORT _variableNames)
# foreach(_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()