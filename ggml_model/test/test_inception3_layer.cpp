#include "inception3.h"
#include "utils.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <assert.h>
#include <random>

using inception::Conv2dLayer;
using inception::DenseLayer;
using inception::Inception3Model;

// TestConv2d 类：用于测试 Conv2d 层
class TestConv2d {
private:
    ggml_context* ctx;        // GGML 上下文
    Conv2dLayer& layer;       // 卷积层参数
    ggml_backend_t backend;   // 计算后端
    int stride_w;             // 宽度方向步长
    int stride_h;             // 高度方向步长
    int padding_w;            // 宽度方向填充
    int padding_h;            // 高度方向填充
    int dilation_w;           // 宽度方向扩张
    int dilation_h;           // 高度方向扩张
    struct ggml_tensor* output; // 输出张量
    std::string layer_name;      // 层名称
    
public:
    // 构造函数
    TestConv2d(ggml_context* ctx, Conv2dLayer& layer, ggml_backend_t backend,
               const char* str_name,
               int stride_w = 1, int stride_h = 1,
               int padding_w = 0, int padding_h = 0,
               int dilation_w = 1, int dilation_h = 1) 
    : ctx(ctx), layer(layer), backend(backend), 
      stride_w(stride_w), stride_h(stride_h),
      padding_w(padding_w), padding_h(padding_h),
      dilation_w(dilation_w), dilation_h(dilation_h),
    output(nullptr) {
        // 设置层名称
        layer_name = str_name;
    }
    
    // 构建计算图
    ggml_cgraph* build_graph() {
        std::string name = layer_name;
        // 创建一个新的计算图
        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        if (!gf) {
            fprintf(stderr, "无法创建计算图\n");
            return nullptr;
        }
        
        // 设置输入尺寸
        int width = 221;
        int height = 100;
        int channels = 7;
        // 创建输入张量
        struct ggml_tensor* input = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, channels, 1);
        name = layer_name + "_input";
        ggml_set_name(input, name.c_str());
        ggml_set_input(input);

        // 执行卷积并构建计算图
        struct ggml_tensor* result = ggml_conv_2d(
            ctx, 
            layer.weights, 
            input, 
            stride_h, stride_w,             // 步长
            padding_h, padding_w,           // 填充
            dilation_h, dilation_w          // 扩张
        );
        name = layer_name + "_conv";
        ggml_set_name(result, name.c_str());
        
        // 批量归一化（如果需要）
        if (layer.batch_normalize) {
            result = ggml_sub(ctx, result, ggml_repeat(ctx, layer.moving_mean, result));
            name = layer_name + "_sub";
            ggml_set_name(result, name.c_str());
            result = ggml_div(ctx, result, ggml_sqrt(ctx, ggml_repeat(ctx, layer.moving_variance, result)));
            name = layer_name + "_div";
            ggml_set_name(result, name.c_str());
            result = ggml_add(ctx, result, ggml_repeat(ctx, layer.beta, result));
            name = layer_name + "_add";
            ggml_set_name(result, name.c_str());
        }

        // 激活函数（如果需要）
        if (layer.activate) {
            result = ggml_relu(ctx, result);
            name = layer_name + "_relu";
            ggml_set_name(result, name.c_str());
        }
        
        // 保存输出张量
        output = result;
        // 设置输出张量的名称
        // ggml_set_name(output, "output");
        
        // 将输出添加到计算图中
        ggml_build_forward_expand(gf, result);
        
        return gf;
    }
    
    // 执行计算
    bool compute(ggml_cgraph* gf) {
        if (!gf) {
            fprintf(stderr, "计算图为空\n");
            return false;
        }
        
        // 执行计算
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "%s: ggml_backend_graph_compute() 失败\n", __func__);
            return false;
        }
        
        return true;
    }
    
    // 实现 apply 方法，构建并执行卷积操作
    ggml_tensor* apply() {
        // 构建计算图
        struct ggml_cgraph* gf = build_graph();
        if (!gf) {
            fprintf(stderr, "构建计算图失败\n");
            return nullptr;
        }
        
        // 执行计算
        if (!compute(gf)) {
            fprintf(stderr, "执行计算失败\n");
            return nullptr;
        }
        
        return output;
    }
    
    // 获取输出张量
    ggml_tensor* get_output() {
        return output;
    }
    
    // 通过名称获取中间结果
    ggml_tensor* get_intermediate_result(const char* name) {
        ggml_tensor* res_tensor = find_tensor_by_name(ctx, name);
        if (!res_tensor) {
            fprintf(stderr, "未找到中间结果: %s\n", name);
            return nullptr;
        }
        return res_tensor;
    }
    
    // 打印所有中间结果
    void print_all_intermediate_results() {
        std::string name = layer_name;
        printf("\n=== 输入张量 ===\n");
        name = layer_name + "_input";
        ggml_tensor* input_tensor = get_intermediate_result(name.c_str());
        if (input_tensor) {
            printf("\n输入张量:\n");
            print_tensor_info(name.c_str(), input_tensor);
        }
        printf("\n=== 输出张量 ===\n");
        name = layer_name + "_output";
        ggml_tensor* output_tensor = get_intermediate_result(name.c_str());
        if (output_tensor) {
            printf("\n输出张量:\n");
            print_tensor_info(name.c_str(), output_tensor);
        }

        printf("\n=== 所有中间计算结果 ===\n");
        // 卷积结果
        name = layer_name + "_conv";
        ggml_tensor* conv_result = get_intermediate_result(name.c_str());
        if (conv_result) {
            printf("\n卷积结果:\n");
            print_tensor_info(name.c_str(), conv_result);
        }
        
        // 批量归一化中间结果
        name = layer_name + "_sub";
        ggml_tensor* sub_result = get_intermediate_result(name.c_str());
        if (sub_result) {
            printf("\n去均值结果:\n");
            print_tensor_info(name.c_str(), sub_result);
        }
        
        name = layer_name + "_div";
        ggml_tensor* div_result = get_intermediate_result(name.c_str());
        if (div_result) {
            printf("\n标准差:\n");
            print_tensor_info(name.c_str(), div_result);
        }
        
        name = layer_name + "_add";
        ggml_tensor* add_result = get_intermediate_result(name.c_str());
        if (add_result) {
            printf("\n批量归一化结果:\n");
            print_tensor_info(name.c_str(), add_result);
        }
        
        name = layer_name + "_relu";
        ggml_tensor* relu_result = get_intermediate_result(name.c_str());
        if (relu_result) {
            printf("\nReLU激活结果:\n");
            print_tensor_info(name.c_str(), relu_result);
        }
    }
};

// 测试 Conv2d_1a_3x3 层的函数
bool test_inception_conv2d_1a_layer(const std::string& model_path) {
    printf("正在测试 Conv2d_1a_3x3 层...\n");
    
    // 初始化模型
    Inception3Model model;
    if (!model.load_model(model_path)) {
        fprintf(stderr, "无法加载模型: %s\n", model_path.c_str());
        return false;
    }
    
    // 获取 Conv2d_1a_3x3 层的权重和参数
    Conv2dLayer& conv_layer = model.conv2d_layers[0];
    
    // 创建 GGML 计算用的上下文
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context * ctx_cgraph = ggml_init(params0);
    // 创建TestConv2d类实例
    printf("\n=== 测试 Conv2d_1a_3x3 层 ===\n");
    TestConv2d test_conv2d(ctx_cgraph, conv_layer, model.backend, "1a", 2, 2, 0, 0, 1, 1);
    
    // 构建计算图
    struct ggml_cgraph* gf = test_conv2d.build_graph();
    if (!gf) {
        fprintf(stderr, "构建计算图失败\n");
        ggml_free(ctx_cgraph);
        return false;
    }
    
    // 分配计算资源
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    
    // 设置输入张量，使用随机数据
    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "1a_input");
    // fill_random_input(model);
    float * data0 = new float[model.width * model.height * model.channels];
    for (int i = 0; i < model.width * model.height * model.channels; i++) {
        data0[i] = 0.5f;
    }
    ggml_backend_tensor_set(input, data0, 0, ggml_nbytes(input));
    delete[] data0;

    // 执行计算
    if (!test_conv2d.compute(gf)) {
        fprintf(stderr, "执行计算失败\n");
        ggml_free(ctx_cgraph);
        return false;
    }
    // Dump the cgraph to a dot file
    ggml_graph_dump_dot(gf, NULL, "debug.dot");
    
    // 打印所有中间结果
    test_conv2d.print_all_intermediate_results();
    
    printf("\nConv2d_1a_3x3 层测试完成！\n");
    // 释放计算图和上下文
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);

    return true;
}


int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "用法: %s <model-path> <layer-name>\n", argv[0]);
        fprintf(stderr, "可用的层名称: Conv2d_1a_3x3, Conv2d_2a_3x3, Conv2d_2b_3x3, maxpool1, ..., Mixed_7c, avgpool, fc\n");
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string layer_name = argv[2];
    
    if (layer_name == "Conv2d_1a_3x3") {
        // 使用特殊的测试函数测试第一个卷积层
        if (!test_inception_conv2d_1a_layer(model_path)) {
            fprintf(stderr, "测试失败！\n");
            return 1;
        }
    } else {
        // 使用通用测试函数测试其他层
        if (!test_inception3_layer(model_path, layer_name)) {
            fprintf(stderr, "测试失败！\n");
            return 1;
        }
    }
    
    printf("测试成功完成！\n");
    return 0;
}
