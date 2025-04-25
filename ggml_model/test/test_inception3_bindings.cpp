#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <string>
#include <memory>
#include <dlfcn.h> // 用于动态加载库

// 用于测量时间的辅助函数
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    }
};

// 定义函数指针类型
typedef int (*create_model_fn)();
typedef bool (*load_model_fn)(int, const char*);
typedef bool (*infer_fn)(int, const float*, int, float**, int*);
typedef void (*free_model_fn)(int);
typedef int (*get_model_count_fn)();

// 结构体，保存所有函数指针
struct Inception3API {
    void* handle;
    create_model_fn create_model;
    load_model_fn load_model;
    infer_fn infer;
    free_model_fn free_model;
    get_model_count_fn get_model_count;
    
    // 析构函数用于关闭库
    ~Inception3API() {
        if (handle) {
            dlclose(handle);
        }
    }
};

// 加载动态库和所有函数
std::unique_ptr<Inception3API> load_inception3_lib(const std::string& lib_path) {
    auto api = std::make_unique<Inception3API>();
    
    // 加载动态库
    api->handle = dlopen(lib_path.c_str(), RTLD_LAZY);
    if (!api->handle) {
        std::cerr << "无法加载库: " << dlerror() << std::endl;
        return nullptr;
    }
    
    // 加载函数指针
    api->create_model = (create_model_fn)dlsym(api->handle, "inception3_create_model");
    api->load_model = (load_model_fn)dlsym(api->handle, "inception3_load_model");
    api->infer = (infer_fn)dlsym(api->handle, "inception3_infer");
    api->free_model = (free_model_fn)dlsym(api->handle, "inception3_free_model");
    api->get_model_count = (get_model_count_fn)dlsym(api->handle, "inception3_get_model_count");
    
    // 检查是否所有函数都已成功加载
    const char* error = dlerror();
    if (error || !api->create_model || !api->load_model || 
        !api->infer || !api->free_model || !api->get_model_count) {
        std::cerr << "加载函数失败: " << (error ? error : "未知错误") << std::endl;
        return nullptr;
    }
    
    return api;
}

// 测试单次推理
bool test_single_inference(const std::unique_ptr<Inception3API>& api, 
                          const std::string& model_path, 
                          int batch_size = 1, 
                          bool verbose = true) {
    if (!api) {
        std::cerr << "API未初始化" << std::endl;
        return false;
    }
    
    if (verbose) {
        std::cout << "\n--- 测试单次推理 (批次大小=" << batch_size << ") ---" << std::endl;
    }

    // 创建模型实例
    int model_id = api->create_model();
    
    // 加载模型
    if (verbose) {
        std::cout << "加载模型: " << model_path << std::endl;
    }
    bool success = api->load_model(model_id, model_path.c_str());
    if (!success) {
        std::cerr << "模型加载失败！" << std::endl;
        return false;
    }
    
    // 创建输入数据
    std::vector<float> input_data(batch_size * 7 * 100 * 221, 0.1f);
    
    // 执行推理
    if (verbose) {
        std::cout << "执行推理..." << std::endl;
    }
    
    Timer timer;
    float* result = nullptr;
    int result_size = 0;
    
    success = api->infer(model_id, input_data.data(), batch_size, &result, &result_size);
    
    double inference_time = timer.elapsed_ms();
    
    if (!success || !result) {
        std::cerr << "推理失败！" << std::endl;
        api->free_model(model_id);
        return false;
    }
    
    if (verbose) {
        std::cout << "推理完成，用时: " << inference_time << " ms" << std::endl;
        std::cout << "输出大小: " << result_size << std::endl;
        
        int output_size_per_batch = result_size / batch_size;
        for (int i = 0; i < batch_size; ++i) {
            std::cout << "第" << i << "个样本的部分结果: ";
            for (int j = 0; j < std::min(3, output_size_per_batch); ++j) {
                std::cout << result[i * output_size_per_batch + j] << " ";
            }
            std::cout << "..." << std::endl;
        }
    }
    
    // 释放资源
    api->free_model(model_id);
    
    if (verbose) {
        std::cout << "模型资源已释放" << std::endl;
    }
    
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "用法: " << argv[0] << " <lib_path> <model_path> [batch_size]" << std::endl;
        return 1;
    }
    
    std::string lib_path = argv[1];
    std::string model_path = argv[2];
    int batch_size = (argc > 3) ? std::stoi(argv[3]) : 1;
    
    std::cout << "加载库: " << lib_path << std::endl;
    std::cout << "使用模型: " << model_path << std::endl;
    std::cout << "批次大小: " << batch_size << std::endl;
    
    // 加载库和函数
    auto api = load_inception3_lib(lib_path);
    if (!api) {
        std::cerr << "加载Inception3库失败!" << std::endl;
        return 1;
    }
    
    // 执行测试
    bool result = test_single_inference(api, model_path, batch_size);
    
    // 检查当前模型计数
    int remaining_models = api->get_model_count();
    std::cout << "剩余模型实例数: " << remaining_models << " (应为0)" << std::endl;
    
    std::cout << "\n测试完成！" << (result ? "成功" : "失败") << std::endl;
    
    return result ? 0 : 1;
}
