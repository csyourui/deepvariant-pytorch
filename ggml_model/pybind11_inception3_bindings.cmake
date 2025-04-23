# 添加Python绑定库
pybind11_add_module(inception3_bindings 
    inception3_bindings.cpp
    inception3.cpp
    utils.cpp
)

# 如果使用自定义 ggml，添加源文件到目标
if(NOT USE_SYSTEM_GGML)
    # 为 Apple 平台添加 Metal 相关内容
    if(APPLE)
        target_sources(inception3_bindings PRIVATE ${CMAKE_BINARY_DIR}/ggml-metal.metallib)
        set_source_files_properties(${CMAKE_BINARY_DIR}/ggml-metal.metallib PROPERTIES
            GENERATED TRUE
            MACOSX_PACKAGE_LOCATION Resources
        )
        
        # 告诉编译器我们正在使用 Metal
        target_compile_definitions(inception3_bindings PRIVATE -DGGML_USE_METAL=1)
    endif()
    target_link_libraries(inception3_bindings PRIVATE ggml ggml-base ggml-metal ggml-cpu ggml-blas)
else()
    target_link_libraries(inception3_bindings PRIVATE ${GGML_LIBRARIES})
endif()

# 链接必要的库
target_link_libraries(inception3_bindings PRIVATE Threads::Threads)

# 为 Apple 平台添加特定的库链接
if(APPLE)
    target_link_libraries(inception3_bindings PRIVATE
        ${ACCELERATE_LIBRARY}
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${MODELIO_LIBRARY}
        ${METALKIT_LIBRARY}
    )
endif()

# 如果启用了 OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(inception3_bindings PRIVATE OpenMP::OpenMP_CXX)
endif()

# 设置输出目录为Python可以找到的位置
set_target_properties(inception3_bindings PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

message(STATUS "Python bindings will be built as: inception3_bindings")
