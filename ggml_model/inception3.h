#ifndef INCEPTION3_H
#define INCEPTION3_H

#include <string>
#include <vector>
#include <memory>

// 前向声明
struct ggml_tensor;
struct ggml_context;
struct ggml_cgraph;
typedef struct ggml_backend* ggml_backend_t;
typedef struct ggml_backend_buffer* ggml_backend_buffer_t;

namespace inception {

// 卷积层类
class Conv2dLayer {
public:
    struct ggml_tensor* weights = nullptr;
    struct ggml_tensor* beta = nullptr;
    struct ggml_tensor* moving_mean = nullptr;
    struct ggml_tensor* moving_variance = nullptr;
    float eps = 1e-3f;  // 防止除零, tensorflow: 1e-3 torch: 1e-5
    bool batch_normalize = true;
    bool activate = true;  // true 为 relu, false 为 linear

public:
    // 默认构造函数
    Conv2dLayer() = default;
    
    // 完整参数构造函数
    Conv2dLayer(
        struct ggml_tensor* weights,
        struct ggml_tensor* beta,
        struct ggml_tensor* moving_mean,
        struct ggml_tensor* moving_variance,
        bool batch_normalize = true,
        bool activate = true
    ) : weights(weights), 
        beta(beta), 
        moving_mean(moving_mean),
        moving_variance(moving_variance),
        batch_normalize(batch_normalize),
        activate(activate) {}
};
    
// 全连接层类
class DenseLayer {
public:
    struct ggml_tensor* kernel = nullptr;
    struct ggml_tensor* biases = nullptr;
    bool softmax = true;
    
public:
    // 默认构造函数
    DenseLayer() = default;
    
    // 完整参数构造函数
    DenseLayer(
        struct ggml_tensor* kernel,
        struct ggml_tensor* biases,
        bool softmax = true
    ) : kernel(kernel),
        biases(biases),
        softmax(softmax) {}
};

// Inception3Model 类
class Inception3Model {
public:
    std::string arch = "inception3";
    int width = 221;
    int height = 100;
    int channels = 7;
    struct ggml_tensor* images = nullptr;

    int classes = 3;
    struct ggml_tensor* classification = nullptr;

    std::vector<Conv2dLayer> conv2d_layers;
    std::vector<DenseLayer> dense_layers;

    ggml_backend_t backend = NULL;
    ggml_backend_buffer_t buffer;
    struct ggml_context* ctx;

public:
    // 构造函数
    Inception3Model();
    
    // 析构函数
    ~Inception3Model();
    
    // 加载模型
    bool load_model(const std::string& fname);
    
    // 构建计算图
    struct ggml_cgraph* build_graph();

    // Getters 和 Setters
    ggml_context* get_context() const;
    ggml_backend_t get_backend() const;
    ggml_backend_buffer_t get_buffer() const;
    const std::vector<Conv2dLayer>& get_conv2d_layers() const;
    const std::vector<DenseLayer>& get_dense_layers() const;
    Conv2dLayer& get_conv2d_layer(int idx);
    DenseLayer& get_dense_layer(int idx);
    struct ggml_tensor* get_images() const;
    struct ggml_tensor* get_classification() const;
    
    // 获取模型维度
    int get_width() const;
    int get_height() const;
    int get_channels() const;
    int get_classes() const;
    
    // 设置模型维度
    void set_dimensions(int w, int h, int c);

    // 用于调试的函数，打印指定名称的中间张量
    void print_intermediate_tensor(const char* name);
    
    // 执行推理并打印结果
    bool infer(struct ggml_cgraph * gf);
};

} // namespace inception

// 逐层测试 inception3 模型的函数
bool test_inception3_layer(const std::string& model_path, const std::string& layer_name);

#endif // INCEPTION3_H