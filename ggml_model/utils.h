
#ifndef GGML_MODEL_UTILS_H
#define GGML_MODEL_UTILS_H

#include "ggml.h"

// 通用张量打印函数，支持多种数据类型
void print_tensor_info(const char* name, ggml_tensor* tensor, int max_elements = 0);
ggml_tensor* find_tensor_by_name(ggml_context* ctx, const char* name);

#endif // GGML_MODEL_UTILS_H