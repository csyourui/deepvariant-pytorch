#include "utils.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <assert.h>

// 通用张量打印函数，支持多种数据类型
// 打印张量的辅助函数
void print_tensor_info(const char* name, ggml_tensor* tensor, int max_elements) {
    if (tensor == nullptr) {
        fprintf(stderr, "Tensor '%s' not found\n", name);
        return;
    }
    
    // 打印张量基本信息
    fprintf(stderr, "Tensor '%s':\n", name);
    fprintf(stderr, "  shape: [%lld, %lld, %lld, %lld]\n", 
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    fprintf(stderr, "  type: %d (%s)\n", tensor->type, ggml_type_name(tensor->type));
    
    // 获取张量的总元素数
    size_t nelements = ggml_nelements(tensor);
    fprintf(stderr, "  elements: %zu\n", nelements);
    
    // 根据数据类型打印元素值
    size_t max_elements_to_print = nelements < max_elements ? nelements : max_elements;
    // fprintf(stderr, "  values (first %zu elements): ", max_elements_to_print);
    
    // 创建临时缓冲区用于统计和打印
    std::vector<float> converted_data(nelements);
    
    // 根据数据类型转换值
    switch (tensor->type) {
        case GGML_TYPE_F32: {
            float* data = (float*)ggml_get_data(tensor);
            for (size_t i = 0; i < nelements; i++) {
                converted_data[i] = data[i];
            }
            break;
        }
        case GGML_TYPE_F16: {
            ggml_fp16_t* data = (ggml_fp16_t*)ggml_get_data(tensor);
            for (size_t i = 0; i < nelements; i++) {
                converted_data[i] = ggml_fp16_to_fp32(data[i]);
            }
            break;
        }
        default:
            fprintf(stderr, "Error: Unsupported tensor type for printing: %s\n", 
                   ggml_type_name(tensor->type));
            return;
    }
    
    // 计算统计信息
    float min_val = converted_data[0], max_val = converted_data[0], sum = 0.0f;
    for (size_t i = 0; i < nelements; i++) {
        min_val = std::min(min_val, converted_data[i]);
        max_val = std::max(max_val, converted_data[i]);
        sum += converted_data[i];
    }
    
    float mean = sum / nelements;
    float stddev = 0.0f;
    for (size_t i = 0; i < nelements; i++) {
        stddev += (converted_data[i] - mean) * (converted_data[i] - mean);
    }
    stddev = sqrt(stddev / nelements);
    
    fprintf(stderr, "  min: %f, max: %f, mean: %f stddev: %f\n", 
            min_val, max_val, mean, stddev);
    fprintf(stderr, "  values: ");
    fprintf(stderr, "\n");
    
    // 打印转换后的前几个值
    for (size_t i = 0; i < max_elements_to_print; i++) {
        fprintf(stderr, "%f ", converted_data[i]);
        if (i % 5 == 4 && i < max_elements_to_print - 1) {
            fprintf(stderr, "\n                           ");
        }
    }
    fprintf(stderr, "\n");
}

// 根据名称查找张量
ggml_tensor* find_tensor_by_name(ggml_context* ctx, const char* name) {
    for (struct ggml_tensor* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
        const char* tensor_name = ggml_get_name(t);
        if (tensor_name && strcmp(tensor_name, name) == 0) {
            return t;
        }
    }
    return nullptr;
}

