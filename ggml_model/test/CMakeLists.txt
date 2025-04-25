# 测试代码的CMakeLists.txt
# 设置 CMake 最低版本要求
cmake_minimum_required(VERSION 3.10)
# 设置项目名称
project(test_inception3)
# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加更多 include 目录
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/..)

# 设置 test_inception3 源文件
set(TEST_INECPTION3_SOURCES
    ${CMAKE_SOURCE_DIR}/inception3.cpp
    ${CMAKE_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_inception3.cpp
)

# 设置 test_inception3_layer 源文件
set(TEST_INECPTION3_LAYER_SOURCES
    ${CMAKE_SOURCE_DIR}/inception3.cpp
    ${CMAKE_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_inception3_layer.cpp
)

# 设置 test_inception3_bindings 源文件
set(TEST_INECPTION3_BINDINGS_SOURCES
    ${CMAKE_SOURCE_DIR}/inception3.cpp
    ${CMAKE_SOURCE_DIR}/utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_inception3_bindings.cpp
)

# 创建所有测试目标的列表
set(ALL_TEST_TARGETS 
    test_inception3
    test_inception3_layer
    test_inception3_bindings
)

# 添加所有可执行文件
add_executable(test_inception3 ${TEST_INECPTION3_SOURCES})
add_executable(test_inception3_layer ${TEST_INECPTION3_LAYER_SOURCES})
add_executable(test_inception3_bindings ${TEST_INECPTION3_BINDINGS_SOURCES})

# 如果使用自定义 ggml，添加源文件到所有测试目标
if(NOT USE_SYSTEM_GGML)
    foreach(target ${ALL_TEST_TARGETS})
        target_sources(${target} PRIVATE ${GGML_SOURCES})
        
        # 为 Apple 平台添加 Metal 相关内容
        if(APPLE)
            target_sources(${target} PRIVATE ${CMAKE_BINARY_DIR}/ggml-metal.metallib)
            set_source_files_properties(${CMAKE_BINARY_DIR}/ggml-metal.metallib PROPERTIES
                GENERATED TRUE
                MACOSX_PACKAGE_LOCATION Resources
            )
            
            # 告诉编译器我们正在使用 Metal
            target_compile_definitions(${target} PRIVATE -DGGML_USE_METAL=1)
        endif()
        target_link_libraries(${target} PRIVATE ggml ggml-base ggml-metal ggml-cpu ggml-blas)
    endforeach()
else()
    foreach(target ${ALL_TEST_TARGETS})
        target_link_libraries(${target} PRIVATE ${GGML_LIBRARIES})
    endforeach()
endif()

# 链接必要的库到所有测试目标
foreach(target ${ALL_TEST_TARGETS})
    target_link_libraries(${target} PRIVATE Threads::Threads)
    
    # 为 Apple 平台添加特定的库链接
    if(APPLE)
        target_link_libraries(${target} PRIVATE
            ${ACCELERATE_LIBRARY}
            ${METAL_LIBRARY}
            ${FOUNDATION_LIBRARY}
            ${MODELIO_LIBRARY}
            ${METALKIT_LIBRARY}
        )
    endif()
    
    # 如果启用了 OpenMP
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()
