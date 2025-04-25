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
#include <memory>

#include "utils.h"  // 包含打印张量和时间函数
#include "inception3.h"  // 包含Inception3模型定义
using inception::Inception3Model;

// 为 model.images 创建随机输入
void fill_random_input(Inception3Model& model, int seed = 42) {
    // 初始化随机数生成器
    std::mt19937 rng(seed);  // 使用梅森旋转算法
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // 均匀分布在 [0, 1) 之间
    
    // 获取张量形状和数据
    int64_t width = model.width;
    int64_t height = model.height;
    int64_t channels = model.channels;
    int64_t batch = 1;
    
    // 计算总元素数
    int64_t nelements = width * height * channels * batch;
    
    // 获取指向张量数据的指针
    float* data = (float*)ggml_get_data(model.images);
    
    // 用随机值填充张量
    for (int64_t i = 0; i < nelements; i++) {
        data[i] = dist(rng);
    }
    
    printf("成功创建随机输入: 形状 [%lld, %lld, %lld, %lld]\n", 
           (long long)width, (long long)height, (long long)channels, (long long)batch);
}

// 测试 Inception3 模型
void test_inception3(struct ggml_cgraph* gf, Inception3Model& model, int batch_size, int loop_count) {
    struct ggml_tensor* input = ggml_graph_get_tensor(gf, "images");
    struct ggml_tensor* output = ggml_graph_get_tensor(gf, "classification");
    
    // 设置输入张量，使用随机数据
    // fill_random_input(model);
    float * data0 = new float[ggml_nbytes(input)];
    // 执行推理
    for (int i = 0; i < loop_count; i++) {
        printf("第 %d 次:\t", i);
        for (int j = 0; j < model.width * model.height * model.channels * batch_size; j++) {
            data0[j] = 0.1f * i;
        }
        ggml_backend_tensor_set(input, data0, 0, ggml_nbytes(input));
        const int64_t t_start_ms = ggml_time_ms();
        
        if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "执行计算失败\n");
            return;
        }
        
        const int64_t t_end_ms = ggml_time_ms();
        // printf("推理完成，用时: %lld ms\n", t_end_ms - t_start_ms);

        // 打印输出
        float* output_data = (float*)ggml_get_data(output);
        printf("[%1.6f %1.6f %1.6f]\n", output_data[0], output_data[1], output_data[2]);
    }
    delete[] data0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "用法: %s <model-path> [batch-size] [loop-conut]\n", argv[0]);
        return 1;
    }

    // 设置默认批次大小
    int batch_size = 1;
    int loop_count = 1;
    if (argc < 3) {
        fprintf(stderr, "默认批次大小: %d 默认循环次数: %d\n", batch_size, loop_count);
    } else if (argc == 3) {
        batch_size = atoi(argv[2]);
    } else if (argc == 4) {
        batch_size = atoi(argv[2]);
        loop_count = atoi(argv[3]);
    }
    
    // 创建内存缓冲区用于计算图
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx_cgraph = ggml_init(params0);
    
    // 创建模型实例
    Inception3Model model;
    
    // 加载模型
    printf("正在加载模型: %s\n", argv[1]);
    if (!model.load_model(argv[1])) {
        fprintf(stderr, "加载模型失败: %s\n", argv[1]);
        ggml_free(ctx_cgraph);
        return 1;
    } else {
        printf("模型加载成功\n");
    }
    
    // 构建计算图
    printf("正在构建计算图...\n");
    struct ggml_cgraph* gf = model.build_graph(batch_size);
    if (gf == NULL) {
        fprintf(stderr, "构建计算图失败\n");
        ggml_free(ctx_cgraph);
        return 1;
    }
    printf("计算图构建成功\n");
    
    // 分配计算资源
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    size_t gmem_size = ggml_gallocr_get_buffer_size(allocr, 0);
    printf("计算缓冲区大小: %.2f MB\n", gmem_size / (1024.0 * 1024.0));
    
    // // 打印计算图
    // printf("计算图结构:\n");
    // ggml_graph_print(gf);
    
    // 保存计算图到 dot 文件
    ggml_graph_dump_dot(gf, NULL, "inception3_graph.dot");
    printf("计算图已保存到 inception3_graph.dot\n");
    
    // 测试模型
    test_inception3(gf, model, batch_size, loop_count);
    
    // 释放资源
    printf("正在释放资源...\n");
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    
    printf("程序执行完毕\n");
    
    return 0;
}