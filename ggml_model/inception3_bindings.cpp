#include "inception3.h"
#include "utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <iostream>
#include <map>
#include <mutex>

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

namespace py = pybind11;
using inception::Inception3Model;

// 模型实例管理类，每个实例独立存储所需资源
class Inception3Instance {
private:
    std::unique_ptr<Inception3Model> model;
    ggml_gallocr_t allocr = nullptr;
    struct ggml_cgraph* gf = nullptr;
    struct ggml_context* ctx_cgraph = nullptr;
    bool initialized = false;
    std::string model_path;
    
public:
    // 构造函数
    Inception3Instance() : model(std::make_unique<Inception3Model>()) {}
    
    // 析构函数，释放所有资源
    ~Inception3Instance() {
        free_resources();
    }
    
    // 加载模型并初始化资源
    bool initialize(const std::string& path) {
        // 记录模型路径
        model_path = path;
        
        // 如果已经初始化过，先释放资源
        if (initialized) {
            free_resources();
        }
        
        // 加载模型
        if (!model->load_model(path)) {
            std::cerr << "无法加载模型: " << path << std::endl;
            return false;
        }
        
        // 创建计算图上下文
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);
        if (!ctx_cgraph) {
            std::cerr << "无法创建计算图上下文" << std::endl;
            return false;
        }
        
        // 构建计算图
        gf = model->build_graph(DEFAULT_GRAPH_BATCH_SIZE);
        if (!gf) {
            std::cerr << "无法构建计算图" << std::endl;
            ggml_free(ctx_cgraph);
            ctx_cgraph = nullptr;
            return false;
        }
        
        // 分配计算资源
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model->backend));
        if (!allocr) {
            std::cerr << "无法分配计算资源" << std::endl;
            ggml_free(ctx_cgraph);
            ctx_cgraph = nullptr;
            gf = nullptr;
            return false;
        }
        
        // 为计算图分配内存
        ggml_gallocr_alloc_graph(allocr, gf);
        
        initialized = true;
        return true;
    }
    
    // 执行推理
    py::array_t<float> infer(py::array_t<float> images_array) {
        if (!initialized) {
            throw std::runtime_error("模型未初始化，请先调用initialize方法");
        }
        
        // 检查输入数组的维度
        py::buffer_info buf_info = images_array.request();
        
        if (buf_info.ndim != 4) {
            throw std::runtime_error("输入必须是4维数组 (batch, channel, height, width)");
        }
        
        // 获取维度
        int batch_size = buf_info.shape[0];
        int channels = buf_info.shape[1];
        int height = buf_info.shape[2];
        int width = buf_info.shape[3];
        
        // 检查维度是否符合要求
        if (channels != 7 || height != 100 || width != 221) {
            throw std::runtime_error(
                "输入维度不符合要求，应为 (batch, 7, 100, 221)，实际为 (" + 
                std::to_string(batch_size) + ", " + 
                std::to_string(channels) + ", " + 
                std::to_string(height) + ", " + 
                std::to_string(width) + ")"
            );
        }
        
        std::cout << "正在处理批次大小为 " << batch_size << " 的输入" << std::endl;
        // 创建输出数组
        int n_classes = model->classes;
        py::array_t<float> result = py::array_t<float>({batch_size, n_classes});
        py::buffer_info result_buf = result.request();
        float* result_ptr = static_cast<float*>(result_buf.ptr);
        // 获取结果张量
        struct ggml_tensor* output = model->classification;
        // 获取输出数据
        float* output_data = static_cast<float*>(ggml_get_data(output));

        float* images_data = static_cast<float*>(buf_info.ptr);
        if (images_data == nullptr) {
            throw std::runtime_error("输入数据为空");
        }
        size_t batch_loop_count = batch_size / DEFAULT_GRAPH_BATCH_SIZE + (batch_size % DEFAULT_GRAPH_BATCH_SIZE > 0 ? 1 : 0);
        size_t last_batch_size = batch_size % DEFAULT_GRAPH_BATCH_SIZE;
        for (size_t i = 0; i < batch_loop_count; i++) {
            if (i == batch_loop_count - 1 && last_batch_size > 0) {
                std::cout << "最后一批次大小: " << last_batch_size << std::endl;
                ggml_backend_tensor_set(
                    model->images, 
                    images_data + i * DEFAULT_GRAPH_BATCH_SIZE * channels * height * width,
                    0, 
                    last_batch_size * channels * height * width * sizeof(float)
                );
            } else {
                std::cout << "第 " << i + 1 << " 批次大小: " << DEFAULT_GRAPH_BATCH_SIZE << std::endl;
                ggml_backend_tensor_set(
                    model->images, 
                    images_data + i * DEFAULT_GRAPH_BATCH_SIZE * channels * height * width,
                    0, 
                    DEFAULT_GRAPH_BATCH_SIZE * channels * height * width * sizeof(float)
                );
            }
            if (ggml_backend_graph_compute(model->backend, gf) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("计算图执行失败");
            }
            
            // 复制数据到输出数组
            size_t output_data_offset = i * DEFAULT_GRAPH_BATCH_SIZE * n_classes;
            size_t output_size = (i == batch_loop_count - 1 && last_batch_size > 0) ? last_batch_size : DEFAULT_GRAPH_BATCH_SIZE;
            size_t output_data_size = output_size * n_classes;
            if (output_data == nullptr) {
                throw std::runtime_error("输出数据为空");
            }
            if (output_data_offset + output_data_size > result_buf.size) {
                std::cout << "输出数据超出范围: " << output_data_offset << " + " << output_data_size << " > " << result_buf.size << std::endl;
                throw std::runtime_error("输出数据超出范围");
            }
            memcpy(result_ptr + output_data_offset, output_data, output_data_size * sizeof(float));        }
        
        return result;
    }
    
    // 释放所有资源
    void free_resources() {
        if (allocr) {
            ggml_gallocr_free(allocr);
            allocr = nullptr;
        }
        
        if (ctx_cgraph) {
            ggml_free(ctx_cgraph);
            ctx_cgraph = nullptr;
        }
        
        gf = nullptr;
        initialized = false;
    }
    
    // 返回当前模型是否已初始化
    bool is_initialized() const {
        return initialized;
    }
    
    // 返回模型路径
    std::string get_model_path() const {
        return model_path;
    }
};

// 实例管理器，使用互斥锁保证线程安全
class Inception3Manager {
private:
    std::map<int, std::unique_ptr<Inception3Instance>> instances;
    mutable std::mutex mtx;
    int next_id = 0;
    
public:
    // 创建新实例并返回其ID
    int create_instance() {
        std::lock_guard<std::mutex> lock(mtx);
        int id = next_id++;
        instances[id] = std::make_unique<Inception3Instance>();
        return id;
    }
    
    // 获取指定ID的实例
    Inception3Instance* get_instance(int id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = instances.find(id);
        if (it == instances.end()) {
            return nullptr;
        }
        return it->second.get();
    }
    
    // 删除指定ID的实例
    bool delete_instance(int id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = instances.find(id);
        if (it == instances.end()) {
            return false;
        }
        instances.erase(it);
        return true;
    }
    
    // 获取实例数量
    size_t count() const {
        std::lock_guard<std::mutex> lock(mtx);
        return instances.size();
    }
};

// 全局实例管理器
static Inception3Manager g_manager;

// Python模块定义
PYBIND11_MODULE(inception3_bindings, m) {
    m.doc() = "Inception3 Model Python Interface";
    
    // 创建实例
    m.def("create_model", []() {
        return g_manager.create_instance();
    }, "创建一个新的Inception3模型实例并返回ID");
    
    // 加载模型
    m.def("load_model", [](int model_id, const std::string& model_path) {
        auto instance = g_manager.get_instance(model_id);
        if (!instance) {
            throw std::runtime_error("无效的模型ID: " + std::to_string(model_id));
        }
        return instance->initialize(model_path);
    }, "加载模型并初始化计算资源", py::arg("model_id"), py::arg("model_path"));
    
    // 执行推理
    m.def("infer", [](int model_id, py::array_t<float> images) {
        auto instance = g_manager.get_instance(model_id);
        if (!instance) {
            throw std::runtime_error("无效的模型ID: " + std::to_string(model_id));
        }
        return instance->infer(images);
    }, "使用模型对输入图像进行推理", py::arg("model_id"), py::arg("images"));
    
    // 释放模型资源
    m.def("free_model", [](int model_id) {
        return g_manager.delete_instance(model_id);
    }, "释放模型资源", py::arg("model_id"));
    
    // 获取当前加载的模型数量
    m.def("get_model_count", []() {
        return g_manager.count();
    }, "获取当前加载的模型数量");
    
    // 检查模型是否已初始化
    m.def("is_model_initialized", [](int model_id) {
        auto instance = g_manager.get_instance(model_id);
        if (!instance) {
            return false;
        }
        return instance->is_initialized();
    }, "检查模型是否已初始化", py::arg("model_id"));
    
    // 获取模型路径
    m.def("get_model_path", [](int model_id) {
        auto instance = g_manager.get_instance(model_id);
        if (!instance) {
            throw std::runtime_error("无效的模型ID: " + std::to_string(model_id));
        }
        return instance->get_model_path();
    }, "获取模型路径", py::arg("model_id"));
}
