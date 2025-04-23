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

// 基类：BasicLayer
class BasicLayer {
protected:
    ggml_context* ctx;  // GGML 上下文

public:
    // 构造函数
    BasicLayer(ggml_context* ctx) : ctx(ctx) {}

    // 虚析构函数，确保正确释放派生类资源
    virtual ~BasicLayer() {}

    // 纯虚函数：应用层操作
    virtual ggml_tensor* apply(ggml_tensor* input) = 0;
};
    
// Conv2D 层：继承自 BasicLayer
class BasicConv2d : public BasicLayer {
private:
    Conv2dLayer layer;  // 存储卷积层参数
    int stride_w;       // 宽度方向步长
    int stride_h;       // 高度方向步长
    int padding_w;      // 宽度方向填充
    int padding_h;      // 高度方向填充
    int dilation_w;     // 宽度方向扩张
    int dilation_h;     // 高度方向扩张

public:
    // 构造函数：接收上下文和卷积层参数，以及可选的步长、填充和扩张参数
    BasicConv2d(ggml_context* ctx, const Conv2dLayer& layer, 
                int stride_w, int stride_h,
                int padding_w, int padding_h,
                int dilation_w, int dilation_h) 
        : BasicLayer(ctx), layer(layer), 
          stride_w(stride_w), stride_h(stride_h),
          padding_w(padding_w), padding_h(padding_h),
          dilation_w(dilation_w), dilation_h(dilation_h) {
    }

    // 实现 apply 方法，执行卷积操作
    virtual ggml_tensor* apply(ggml_tensor* input) override {
        // 执行卷积，使用自定义的步长、填充和扩张参数
        struct ggml_tensor* result = ggml_conv_2d(
            ctx, 
            layer.weights, 
            input, 
            stride_h, stride_w,             // 步长
            padding_h, padding_w,           // 填充
            dilation_h, dilation_w          // 扩张
        );

        // 批量归一化（如果需要）
        if (layer.batch_normalize) {
            result = ggml_sub(ctx, result, ggml_repeat(ctx, layer.moving_mean, result));
            result = ggml_div(ctx, result, 
                    ggml_sqrt(ctx, ggml_repeat(ctx, layer.moving_variance, result)));
            result = ggml_add(ctx, result, ggml_repeat(ctx, layer.beta, result));
        }

        // 激活函数（如果需要）
        if (layer.activate) {
            result = ggml_relu(ctx, result);
        }

        return result;
    }

    // 获取卷积层参数的方法
    const Conv2dLayer& get_layer() const { return layer; }
};
    
// 全连接层：继承自 BasicLayer
class BasicDense : public BasicLayer {
private:
    DenseLayer layer;  // 存储全连接层参数

public:
    // 构造函数：接收上下文和全连接层参数
    BasicDense(ggml_context* ctx, const DenseLayer& layer)
        : BasicLayer(ctx), layer(layer) {}

    // 实现 apply 方法，执行全连接操作
    virtual ggml_tensor* apply(ggml_tensor* input) override {
        // 重塑输入为 2D
        struct ggml_tensor* result = ggml_reshape_2d(ctx, input, input->ne[2], input->ne[3]);
        
        // 矩阵乘法
        result = ggml_mul_mat(ctx, layer.kernel, result);
        
        // 添加偏置
        result = ggml_add(ctx, result, layer.biases);
        
        // Softmax（如果需要）
        if (layer.softmax) {
            result = ggml_soft_max(ctx, result);
        }
        
        return result;
    }

    // 获取全连接层参数的方法
    const DenseLayer& get_layer() const { return layer; }
};

// InceptionA 模块：继承自 BasicLayer
class InceptionA : public BasicLayer {
private:
    std::unique_ptr<BasicConv2d> branch1x1;
    std::unique_ptr<BasicConv2d> branch5x5_1;
    std::unique_ptr<BasicConv2d> branch5x5_2;
    std::unique_ptr<BasicConv2d> branch3x3dbl_1;
    std::unique_ptr<BasicConv2d> branch3x3dbl_2;
    std::unique_ptr<BasicConv2d> branch3x3dbl_3;
    std::unique_ptr<BasicConv2d> branch_pool;
    
public:
    // 构造函数：接收上下文和各分支的卷积层参数
    InceptionA(
        ggml_context* ctx,
        const Conv2dLayer& branch1x1_layer,
        const Conv2dLayer& branch5x5_1_layer,
        const Conv2dLayer& branch5x5_2_layer,
        const Conv2dLayer& branch3x3dbl_1_layer,
        const Conv2dLayer& branch3x3dbl_2_layer,
        const Conv2dLayer& branch3x3dbl_3_layer,
        const Conv2dLayer& branch_pool_layer
    ) : BasicLayer(ctx) {
        // 初始化各个分支
        branch1x1 = std::make_unique<BasicConv2d>(ctx, branch1x1_layer, 1, 1, 0, 0, 1, 1);
        branch5x5_1 = std::make_unique<BasicConv2d>(ctx, branch5x5_1_layer, 1, 1, 0, 0, 1, 1);
        branch5x5_2 = std::make_unique<BasicConv2d>(ctx, branch5x5_2_layer, 1, 1, 2, 2, 1, 1);
        branch3x3dbl_1 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_1_layer, 1, 1, 0, 0, 1, 1);
        branch3x3dbl_2 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_2_layer, 1, 1, 1, 1, 1, 1);
        branch3x3dbl_3 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_3_layer, 1, 1, 1, 1, 1, 1);
        branch_pool = std::make_unique<BasicConv2d>(ctx, branch_pool_layer, 1, 1, 0, 0, 1, 1);
    }

    // 替代构造函数：使用指定的卷积层索引从模型中获取参数
    InceptionA(
        ggml_context* ctx,
        Inception3Model& model,
        int branch1x1_idx,
        int branch5x5_1_idx,
        int branch5x5_2_idx,
        int branch3x3dbl_1_idx,
        int branch3x3dbl_2_idx,
        int branch3x3dbl_3_idx,
        int branch_pool_idx
    ) : BasicLayer(ctx) {
        // 从模型中获取相应的卷积层，参考Python版本添加恰当的参数
        // 1x1 卷积，默认步长和无填充
        branch1x1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch1x1_idx], 1, 1, 0, 0, 1, 1);
        
        // 1x1 卷积，默认步长和无填充
        branch5x5_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch5x5_1_idx], 1, 1, 0, 0, 1, 1);
        // 5x5 卷积，默认步长和填充2
        branch5x5_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch5x5_2_idx], 1, 1, 2, 2, 1, 1);
        
        // 1x1 卷积，默认步长和无填充
        branch3x3dbl_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_1_idx], 1, 1, 0, 0, 1, 1);
        // 3x3 卷积，默认步长和填充1
        branch3x3dbl_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_2_idx], 1, 1, 1, 1, 1, 1);
        // 3x3 卷积，默认步长和填充1
        branch3x3dbl_3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_3_idx], 1, 1, 1, 1, 1, 1);
        
        // 1x1 卷积，用于池化分支
        branch_pool = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch_pool_idx], 1, 1, 0, 0, 1, 1);
    }
    
    // 应用 InceptionA 模块的前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // 分支 1x1
        struct ggml_tensor* branch1x1_output = branch1x1->apply(input);
        
        // 分支 5x5
        struct ggml_tensor* branch5x5 = branch5x5_1->apply(input);
        branch5x5 = branch5x5_2->apply(branch5x5);
        
        // 分支 3x3 双重
        struct ggml_tensor* branch3x3dbl = branch3x3dbl_1->apply(input);
        branch3x3dbl = branch3x3dbl_2->apply(branch3x3dbl);
        branch3x3dbl = branch3x3dbl_3->apply(branch3x3dbl);
        
        // 分支池化
        struct ggml_tensor* branch_pool_avg = ggml_pool_2d(
            ctx, input, GGML_OP_POOL_AVG, 
            3, 3,  // kernel size
            1, 1,  // stride
            1, 1   // padding
        );
        struct ggml_tensor* branch_pool_output = branch_pool->apply(branch_pool_avg);
        
        // 拼接所有分支的输出
        // 注：GGML 中的拼接是沿着通道维度（axis=2）
        struct ggml_tensor* concat1 = ggml_concat(ctx, branch1x1_output, branch5x5, 2);
        struct ggml_tensor* concat2 = ggml_concat(ctx, concat1, branch3x3dbl, 2);
        struct ggml_tensor* output = ggml_concat(ctx, concat2, branch_pool_output, 2);
        
        return output;
    }
};

// InceptionB 模块：继承自 BasicLayer
class InceptionB : public BasicLayer {
private:
    std::unique_ptr<BasicConv2d> branch3x3;
    std::unique_ptr<BasicConv2d> branch3x3dbl_1;
    std::unique_ptr<BasicConv2d> branch3x3dbl_2;
    std::unique_ptr<BasicConv2d> branch3x3dbl_3;
    
public:
    // 构造函数：接收上下文和各分支的卷积层参数
    InceptionB(
        ggml_context* ctx,
        const Conv2dLayer& branch3x3_layer,
        const Conv2dLayer& branch3x3dbl_1_layer,
        const Conv2dLayer& branch3x3dbl_2_layer,
        const Conv2dLayer& branch3x3dbl_3_layer
    ) : BasicLayer(ctx) {
        // 初始化各个分支
        branch3x3 = std::make_unique<BasicConv2d>(ctx, branch3x3_layer, 2, 2, 0, 0, 1, 1);
        branch3x3dbl_1 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_1_layer, 1, 1, 0, 0, 1, 1);
        branch3x3dbl_2 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_2_layer, 1, 1, 1, 1, 1, 1);
        branch3x3dbl_3 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_3_layer, 2, 2, 0, 0, 1, 1);
    }

    // 替代构造函数：使用指定的卷积层索引从模型中获取参数
    InceptionB(
        ggml_context* ctx,
        Inception3Model& model,
        int branch3x3_idx,
        int branch3x3dbl_1_idx,
        int branch3x3dbl_2_idx,
        int branch3x3dbl_3_idx
    ) : BasicLayer(ctx) {
        // 从模型中获取相应的卷积层
        // 3x3 卷积，步长为2，无填充
        branch3x3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3_idx], 2, 2, 0, 0, 1, 1);
        
        // 1x1 卷积，默认步长和无填充
        branch3x3dbl_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_1_idx], 1, 1, 0, 0, 1, 1);
        // 3x3 卷积，默认步长和填充1
        branch3x3dbl_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_2_idx], 1, 1, 1, 1, 1, 1);
        // 3x3 卷积，步长为2，无填充
        branch3x3dbl_3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_3_idx], 2, 2, 0, 0, 1, 1);
    }
    
    // 应用 InceptionB 模块的前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // 分支 3x3
        struct ggml_tensor* branch3x3_output = branch3x3->apply(input);
        
        // 分支 3x3dbl
        struct ggml_tensor* branch3x3dbl = branch3x3dbl_1->apply(input);
        branch3x3dbl = branch3x3dbl_2->apply(branch3x3dbl);
        branch3x3dbl = branch3x3dbl_3->apply(branch3x3dbl);
        
        // 分支池化
        struct ggml_tensor* branch_pool = ggml_pool_2d(
            ctx, input, GGML_OP_POOL_MAX,
            3, 3,  // kernel size
            2, 2,  // stride
            0, 0   // padding
        );
        
        // 拼接所有分支的输出
        struct ggml_tensor* concat1 = ggml_concat(ctx, branch3x3_output, branch3x3dbl, 2);
        struct ggml_tensor* output = ggml_concat(ctx, concat1, branch_pool, 2);
        
        return output;
    }
};

// InceptionC 模块：继承自 BasicLayer
class InceptionC : public BasicLayer {
private:
    std::unique_ptr<BasicConv2d> branch1x1;
    std::unique_ptr<BasicConv2d> branch7x7_1;
    std::unique_ptr<BasicConv2d> branch7x7_2;
    std::unique_ptr<BasicConv2d> branch7x7_3;
    std::unique_ptr<BasicConv2d> branch7x7dbl_1;
    std::unique_ptr<BasicConv2d> branch7x7dbl_2;
    std::unique_ptr<BasicConv2d> branch7x7dbl_3;
    std::unique_ptr<BasicConv2d> branch7x7dbl_4;
    std::unique_ptr<BasicConv2d> branch7x7dbl_5;
    std::unique_ptr<BasicConv2d> branch_pool;
    
public:
    // 构造函数：接收上下文和各分支的卷积层参数
    InceptionC(
        ggml_context* ctx,
        const Conv2dLayer& branch1x1_layer,
        const Conv2dLayer& branch7x7_1_layer,
        const Conv2dLayer& branch7x7_2_layer,
        const Conv2dLayer& branch7x7_3_layer,
        const Conv2dLayer& branch7x7dbl_1_layer,
        const Conv2dLayer& branch7x7dbl_2_layer,
        const Conv2dLayer& branch7x7dbl_3_layer,
        const Conv2dLayer& branch7x7dbl_4_layer,
        const Conv2dLayer& branch7x7dbl_5_layer,
        const Conv2dLayer& branch_pool_layer
    ) : BasicLayer(ctx) {
        // 初始化各个分支
        branch1x1 = std::make_unique<BasicConv2d>(ctx, branch1x1_layer, 1, 1, 0, 0, 1, 1);
        branch7x7_1 = std::make_unique<BasicConv2d>(ctx, branch7x7_1_layer, 1, 1, 0, 0, 1, 1);
        branch7x7_2 = std::make_unique<BasicConv2d>(ctx, branch7x7_2_layer, 1, 1, 0, 3, 1, 1);
        branch7x7_3 = std::make_unique<BasicConv2d>(ctx, branch7x7_3_layer, 1, 1, 3, 0, 1, 1);
        branch7x7dbl_1 = std::make_unique<BasicConv2d>(ctx, branch7x7dbl_1_layer, 1, 1, 0, 0, 1, 1);
        branch7x7dbl_2 = std::make_unique<BasicConv2d>(ctx, branch7x7dbl_2_layer, 1, 1, 3, 0, 1, 1);
        branch7x7dbl_3 = std::make_unique<BasicConv2d>(ctx, branch7x7dbl_3_layer, 1, 1, 0, 3, 1, 1);
        branch7x7dbl_4 = std::make_unique<BasicConv2d>(ctx, branch7x7dbl_4_layer, 1, 1, 3, 0, 1, 1);
        branch7x7dbl_5 = std::make_unique<BasicConv2d>(ctx, branch7x7dbl_5_layer, 1, 1, 0, 3, 1, 1);
        branch_pool = std::make_unique<BasicConv2d>(ctx, branch_pool_layer, 1, 1, 0, 0, 1, 1);
    }

    // 替代构造函数：使用指定的卷积层索引从模型中获取参数
    InceptionC(
        ggml_context* ctx,
        Inception3Model& model,
        int branch1x1_idx,
        int branch7x7_1_idx,
        int branch7x7_2_idx,
        int branch7x7_3_idx,
        int branch7x7dbl_1_idx,
        int branch7x7dbl_2_idx,
        int branch7x7dbl_3_idx,
        int branch7x7dbl_4_idx,
        int branch7x7dbl_5_idx,
        int branch_pool_idx
    ) : BasicLayer(ctx) {
        // 从模型中获取相应的卷积层
        // 1x1 卷积，默认步长和无填充
        branch1x1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch1x1_idx], 1, 1, 0, 0, 1, 1);
        
        // 1x1 卷积，默认步长和无填充
        branch7x7_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7_1_idx], 1, 1, 0, 0, 1, 1);
        // 1x7 卷积，默认步长，需要水平方向填充3
        branch7x7_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7_2_idx], 
                                                 1, 1, 0, 3, 1, 1);
        // 7x1 卷积，默认步长，需要垂直方向填充3
        branch7x7_3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7_3_idx], 
                                                 1, 1, 3, 0, 1, 1);
        
        // 1x1 卷积，默认步长和无填充
        branch7x7dbl_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7dbl_1_idx], 1, 1, 0, 0, 1, 1);
        // 7x1 卷积，默认步长，需要垂直方向填充3
        branch7x7dbl_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7dbl_2_idx], 
                                                    1, 1, 3, 0, 1, 1);
        // 1x7 卷积，默认步长，需要水平方向填充3
        branch7x7dbl_3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7dbl_3_idx], 
                                                    1, 1, 0, 3, 1, 1);
        // 7x1 卷积，默认步长，需要垂直方向填充3
        branch7x7dbl_4 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7dbl_4_idx], 
                                                    1, 1, 3, 0, 1, 1);
        // 1x7 卷积，默认步长，需要水平方向填充3
        branch7x7dbl_5 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7dbl_5_idx], 
                                                    1, 1, 0, 3, 1, 1);
        
        // 1x1 卷积，池化分支
        branch_pool = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch_pool_idx], 1, 1, 0, 0, 1, 1);
    }
    
    // 应用 InceptionC 模块的前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // 分支 1x1
        struct ggml_tensor* branch1x1_output = branch1x1->apply(input);
        
        // 分支 7x7：1x1 -> 1x7 -> 7x1
        struct ggml_tensor* branch7x7 = branch7x7_1->apply(input);
        branch7x7 = branch7x7_2->apply(branch7x7); // 应用 1x7 卷积
        branch7x7 = branch7x7_3->apply(branch7x7); // 应用 7x1 卷积
        
        // 分支 7x7dbl：1x1 -> 7x1 -> 1x7 -> 7x1 -> 1x7
        struct ggml_tensor* branch7x7dbl = branch7x7dbl_1->apply(input);
        branch7x7dbl = branch7x7dbl_2->apply(branch7x7dbl); // 应用第一个 7x1 卷积
        branch7x7dbl = branch7x7dbl_3->apply(branch7x7dbl); // 应用第一个 1x7 卷积
        branch7x7dbl = branch7x7dbl_4->apply(branch7x7dbl); // 应用第二个 7x1 卷积
        branch7x7dbl = branch7x7dbl_5->apply(branch7x7dbl); // 应用第二个 1x7 卷积
        
        // 分支池化：平均池化 -> 1x1 卷积
        struct ggml_tensor* branch_pool_avg = ggml_pool_2d(
            ctx, input, GGML_OP_POOL_AVG, 
            3, 3,  // kernel size
            1, 1,  // stride
            1, 1   // padding
        );
        struct ggml_tensor* branch_pool_output = branch_pool->apply(branch_pool_avg);
        
        // 拼接所有分支的输出
        struct ggml_tensor* concat1 = ggml_concat(ctx, branch1x1_output, branch7x7, 2);
        struct ggml_tensor* concat2 = ggml_concat(ctx, concat1, branch7x7dbl, 2);
        struct ggml_tensor* output = ggml_concat(ctx, concat2, branch_pool_output, 2);
        
        return output;
    }
};

// InceptionD 模块：继承自 BasicLayer
class InceptionD : public BasicLayer {
private:
    std::unique_ptr<BasicConv2d> branch3x3_1;
    std::unique_ptr<BasicConv2d> branch3x3_2;
    std::unique_ptr<BasicConv2d> branch7x7x3_1;
    std::unique_ptr<BasicConv2d> branch7x7x3_2;
    std::unique_ptr<BasicConv2d> branch7x7x3_3;
    std::unique_ptr<BasicConv2d> branch7x7x3_4;
    
public:
    // 构造函数：接收上下文和各分支的卷积层参数
    InceptionD(
        ggml_context* ctx,
        const Conv2dLayer& branch3x3_1_layer,
        const Conv2dLayer& branch3x3_2_layer,
        const Conv2dLayer& branch7x7x3_1_layer,
        const Conv2dLayer& branch7x7x3_2_layer,
        const Conv2dLayer& branch7x7x3_3_layer,
        const Conv2dLayer& branch7x7x3_4_layer
    ) : BasicLayer(ctx) {
        // 初始化各个分支
        branch3x3_1 = std::make_unique<BasicConv2d>(ctx, branch3x3_1_layer, 1, 1, 0, 0, 1, 1);
        branch3x3_2 = std::make_unique<BasicConv2d>(ctx, branch3x3_2_layer, 2, 2, 0, 0, 1, 1);
        branch7x7x3_1 = std::make_unique<BasicConv2d>(ctx, branch7x7x3_1_layer, 1, 1, 0, 0, 1, 1);
        branch7x7x3_2 = std::make_unique<BasicConv2d>(ctx, branch7x7x3_2_layer, 1, 1, 0, 3, 1, 1);
        branch7x7x3_3 = std::make_unique<BasicConv2d>(ctx, branch7x7x3_3_layer, 1, 1, 3, 0, 1, 1);
        branch7x7x3_4 = std::make_unique<BasicConv2d>(ctx, branch7x7x3_4_layer, 2, 2, 0, 0, 1, 1);
    }

    // 替代构造函数：使用指定的卷积层索引从模型中获取参数
    InceptionD(
        ggml_context* ctx,
        Inception3Model& model,
        int branch3x3_1_idx,
        int branch3x3_2_idx,
        int branch7x7x3_1_idx,
        int branch7x7x3_2_idx,
        int branch7x7x3_3_idx,
        int branch7x7x3_4_idx
    ) : BasicLayer(ctx) {
        // 从模型中获取相应的卷积层
        // 1x1 卷积，默认步长无填充
        branch3x3_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3_1_idx], 1, 1, 0, 0, 1, 1);
        // 3x3 卷积，步长为2，无填充
        branch3x3_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3_2_idx], 2, 2, 0, 0, 1, 1);
        
        // 1x1 卷积，默认步长无填充
        branch7x7x3_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7x3_1_idx], 1, 1, 0, 0, 1, 1);
        // 1x7 卷积，默认步长，水平方向填充3
        branch7x7x3_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7x3_2_idx], 
                                                   1, 1, 0, 3, 1, 1);
        // 7x1 卷积，默认步长，垂直方向填充3
        branch7x7x3_3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7x3_3_idx], 
                                                   1, 1, 3, 0, 1, 1);
        // 3x3 卷积，步长为2，无填充
        branch7x7x3_4 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch7x7x3_4_idx], 2, 2, 0, 0, 1, 1);
    }
    
    // 应用 InceptionD 模块的前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // 分支 3x3
        struct ggml_tensor* branch3x3 = branch3x3_1->apply(input);
        branch3x3 = branch3x3_2->apply(branch3x3); // 带步长的 3x3 卷积
        
        // 分支 7x7x3
        struct ggml_tensor* branch7x7x3 = branch7x7x3_1->apply(input);
        branch7x7x3 = branch7x7x3_2->apply(branch7x7x3); // 应用 1x7 卷积
        branch7x7x3 = branch7x7x3_3->apply(branch7x7x3); // 应用 7x1 卷积
        branch7x7x3 = branch7x7x3_4->apply(branch7x7x3); // 应用带步长的 3x3 卷积
        
        // 分支池化：最大池化，步长为 2
        struct ggml_tensor* branch_pool = ggml_pool_2d(
            ctx, input, GGML_OP_POOL_MAX, 
            3, 3,  // kernel size
            2, 2,  // stride
            0, 0   // padding
        );
        
        // 拼接所有分支的输出
        struct ggml_tensor* concat1 = ggml_concat(ctx, branch3x3, branch7x7x3, 2);
        struct ggml_tensor* output = ggml_concat(ctx, concat1, branch_pool, 2);
        
        return output;
    }
};

// InceptionE 模块：继承自 BasicLayer
class InceptionE : public BasicLayer {
private:
    std::unique_ptr<BasicConv2d> branch1x1;
    std::unique_ptr<BasicConv2d> branch3x3_1;
    std::unique_ptr<BasicConv2d> branch3x3_2a;
    std::unique_ptr<BasicConv2d> branch3x3_2b;
    std::unique_ptr<BasicConv2d> branch3x3dbl_1;
    std::unique_ptr<BasicConv2d> branch3x3dbl_2;
    std::unique_ptr<BasicConv2d> branch3x3dbl_3a;
    std::unique_ptr<BasicConv2d> branch3x3dbl_3b;
    std::unique_ptr<BasicConv2d> branch_pool;
    
public:
    // 构造函数：接收上下文和各分支的卷积层参数
    InceptionE(
        ggml_context* ctx,
        const Conv2dLayer& branch1x1_layer,
        const Conv2dLayer& branch3x3_1_layer,
        const Conv2dLayer& branch3x3_2a_layer,
        const Conv2dLayer& branch3x3_2b_layer,
        const Conv2dLayer& branch3x3dbl_1_layer,
        const Conv2dLayer& branch3x3dbl_2_layer,
        const Conv2dLayer& branch3x3dbl_3a_layer,
        const Conv2dLayer& branch3x3dbl_3b_layer,
        const Conv2dLayer& branch_pool_layer
    ) : BasicLayer(ctx) {
        // 初始化各个分支
        branch1x1 = std::make_unique<BasicConv2d>(ctx, branch1x1_layer, 1, 1, 0, 0, 1, 1);
        branch3x3_1 = std::make_unique<BasicConv2d>(ctx, branch3x3_1_layer, 1, 1, 0, 0, 1, 1);
        branch3x3_2a = std::make_unique<BasicConv2d>(ctx, branch3x3_2a_layer, 1, 1, 0, 1, 1, 1);
        branch3x3_2b = std::make_unique<BasicConv2d>(ctx, branch3x3_2b_layer, 1, 1, 1, 0, 1, 1);
        branch3x3dbl_1 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_1_layer, 1, 1, 0, 0, 1, 1);
        branch3x3dbl_2 = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_2_layer, 1, 1, 1, 1, 1, 1);
        branch3x3dbl_3a = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_3a_layer, 1, 1, 0, 1, 1, 1);
        branch3x3dbl_3b = std::make_unique<BasicConv2d>(ctx, branch3x3dbl_3b_layer, 1, 1, 1, 0, 1, 1);
        branch_pool = std::make_unique<BasicConv2d>(ctx, branch_pool_layer, 1, 1, 0, 0, 1, 1);
    }

    // 替代构造函数：使用指定的卷积层索引从模型中获取参数
    InceptionE(
        ggml_context* ctx,
        Inception3Model& model,
        int branch1x1_idx,
        int branch3x3_1_idx,
        int branch3x3_2a_idx,
        int branch3x3_2b_idx,
        int branch3x3dbl_1_idx,
        int branch3x3dbl_2_idx,
        int branch3x3dbl_3a_idx,
        int branch3x3dbl_3b_idx,
        int branch_pool_idx
    ) : BasicLayer(ctx) {
        // 从模型中获取相应的卷积层
        // 1x1 卷积，默认步长无填充
        branch1x1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch1x1_idx], 1, 1, 0, 0, 1, 1);
        
        // 1x1 卷积，默认步长无填充
        branch3x3_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3_1_idx], 1, 1, 0, 0, 1, 1);
        // 1x3 卷积，默认步长，水平方向填充1
        branch3x3_2a = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3_2a_idx], 
                                                  1, 1, 0, 1, 1, 1);
        // 3x1 卷积，默认步长，垂直方向填充1
        branch3x3_2b = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3_2b_idx], 
                                                  1, 1, 1, 0, 1, 1);
        
        // 1x1 卷积，默认步长无填充
        branch3x3dbl_1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_1_idx], 1, 1, 0, 0, 1, 1);
        // 3x3 卷积，默认步长，填充1
        branch3x3dbl_2 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_2_idx], 1, 1, 1, 1, 1, 1);
        // 1x3 卷积，默认步长，水平方向填充1
        branch3x3dbl_3a = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_3a_idx], 
                                                     1, 1, 0, 1, 1, 1);
        // 3x1 卷积，默认步长，垂直方向填充1
        branch3x3dbl_3b = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch3x3dbl_3b_idx], 
                                                     1, 1, 1, 0, 1, 1);
        
        // 1x1 卷积，池化分支
        branch_pool = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[branch_pool_idx], 1, 1, 0, 0, 1, 1);
    }
    
    // 应用 InceptionE 模块的前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // 分支 1x1
        struct ggml_tensor* branch1x1_output = branch1x1->apply(input);
        
        // 分支 3x3
        struct ggml_tensor* branch3x3 = branch3x3_1->apply(input);
        struct ggml_tensor* branch3x3_a = branch3x3_2a->apply(branch3x3);
        struct ggml_tensor* branch3x3_b = branch3x3_2b->apply(branch3x3);
        // 拼接 1x3 和 3x1 卷积的结果
        struct ggml_tensor* branch3x3_output = ggml_concat(ctx, branch3x3_a, branch3x3_b, 2);
        
        // 分支 3x3dbl
        struct ggml_tensor* branch3x3dbl = branch3x3dbl_1->apply(input);
        branch3x3dbl = branch3x3dbl_2->apply(branch3x3dbl);
        struct ggml_tensor* branch3x3dbl_a = branch3x3dbl_3a->apply(branch3x3dbl);
        struct ggml_tensor* branch3x3dbl_b = branch3x3dbl_3b->apply(branch3x3dbl);
        // 拼接 1x3 和 3x1 卷积的结果
        struct ggml_tensor* branch3x3dbl_output = ggml_concat(ctx, branch3x3dbl_a, branch3x3dbl_b, 2);
        
        // 分支池化：平均池化 -> 1x1 卷积
        struct ggml_tensor* branch_pool_avg = ggml_pool_2d(
            ctx, input, GGML_OP_POOL_AVG, 
            3, 3,  // kernel size
            1, 1,  // stride
            1, 1   // padding
        );
        struct ggml_tensor* branch_pool_output = branch_pool->apply(branch_pool_avg);
        
        // 拼接所有分支的输出
        struct ggml_tensor* concat1 = ggml_concat(ctx, branch1x1_output, branch3x3_output, 2);
        struct ggml_tensor* concat2 = ggml_concat(ctx, concat1, branch3x3dbl_output, 2);
        struct ggml_tensor* output = ggml_concat(ctx, concat2, branch_pool_output, 2);
        
        return output;
    }
};

// InceptionAux 模块：继承自 BasicLayer
class InceptionAux : public BasicLayer {
private:
    std::unique_ptr<BasicConv2d> conv0;
    std::unique_ptr<BasicConv2d> conv1;
    std::unique_ptr<BasicDense> fc;
    
public:
    // 构造函数：接收上下文和各层参数
    InceptionAux(
        ggml_context* ctx,
        const Conv2dLayer& conv0_layer,
        const Conv2dLayer& conv1_layer,
        const DenseLayer& fc_layer
    ) : BasicLayer(ctx) {
        // 初始化各层
        conv0 = std::make_unique<BasicConv2d>(ctx, conv0_layer, 1, 1, 0, 0, 1, 1);
        conv1 = std::make_unique<BasicConv2d>(ctx, conv1_layer, 1, 1, 0, 0, 1, 1);
        fc = std::make_unique<BasicDense>(ctx, fc_layer);
    }

    // 替代构造函数：使用指定的层索引从模型中获取参数
    InceptionAux(
        ggml_context* ctx,
        Inception3Model& model,
        int conv0_idx,
        int conv1_idx,
        int fc_idx
    ) : BasicLayer(ctx) {
        // 从模型中获取相应的层
        // 1x1 卷积，默认步长无填充
        conv0 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[conv0_idx], 1, 1, 0, 0, 1, 1);
        // 5x5 卷积，默认步长无填充
        conv1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[conv1_idx], 1, 1, 0, 0, 1, 1);
        fc = std::make_unique<BasicDense>(ctx, model.dense_layers[fc_idx]);
    }
    
    // 应用 InceptionAux 模块的前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // 平均池化，步长为 3，核大小为 5x5
        struct ggml_tensor* result = ggml_pool_2d(
            ctx, input, GGML_OP_POOL_AVG,
            5, 5,  // kernel size
            3, 3,  // stride
            0, 0   // padding
        );
        
        // 应用 1x1 卷积
        result = conv0->apply(result);
        
        // 应用 5x5 卷积
        result = conv1->apply(result);
        
        // 全局平均池化
        result = ggml_pool_2d(
            ctx, result, GGML_OP_POOL_AVG,
            result->ne[0], result->ne[1],  // kernel size = feature map size
            1, 1,  // stride
            0, 0   // padding
        );
        
        // 应用全连接层
        result = fc->apply(result);
        
        return result;
    }
};

// Inception3 类: 整个网络架构
class Inception3 : public BasicLayer {
public:
    // Stem 部分（最初的卷积层）
    std::unique_ptr<BasicConv2d> Conv2d_1a_3x3;  // 第一卷积层
    std::unique_ptr<BasicConv2d> Conv2d_2a_3x3;  // 第二卷积层
    std::unique_ptr<BasicConv2d> Conv2d_2b_3x3;  // 第三卷积层
    std::unique_ptr<BasicConv2d> Conv2d_3b_1x1;  // 第四卷积层
    std::unique_ptr<BasicConv2d> Conv2d_4a_3x3;  // 第五卷积层
    
    // Inception 模块
    std::unique_ptr<InceptionA> Mixed_5b;
    std::unique_ptr<InceptionA> Mixed_5c;
    std::unique_ptr<InceptionA> Mixed_5d;
    std::unique_ptr<InceptionB> Mixed_6a;
    std::unique_ptr<InceptionC> Mixed_6b;
    std::unique_ptr<InceptionC> Mixed_6c;
    std::unique_ptr<InceptionC> Mixed_6d;
    std::unique_ptr<InceptionC> Mixed_6e;
    std::unique_ptr<InceptionD> Mixed_7a;
    std::unique_ptr<InceptionE> Mixed_7b;
    std::unique_ptr<InceptionE> Mixed_7c;
    
    // 最终分类层
    std::unique_ptr<BasicDense> fc;
    
    // 是否使用辅助分类器
    bool use_aux_logits;
    std::unique_ptr<InceptionAux> AuxLogits;
    
public:
    // 构造函数：使用模型的索引初始化各层
    Inception3(
        ggml_context* ctx,
        Inception3Model& model,
        bool use_aux_logits = false
    ) : BasicLayer(ctx), use_aux_logits(use_aux_logits) {
        // 初始化 Stem 部分，参考Python版本添加适当的参数
        Conv2d_1a_3x3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[0], 2, 2, 0, 0, 1, 1); // 步长为2，无填充
        Conv2d_2a_3x3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[1], 1, 1, 0, 0, 1, 1); // 默认步长，无填充
        Conv2d_2b_3x3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[2], 1, 1, 1, 1, 1, 1); // 默认步长，填充1
        Conv2d_3b_1x1 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[3], 1, 1, 0, 0, 1, 1); // 默认步长，无填充
        Conv2d_4a_3x3 = std::make_unique<BasicConv2d>(ctx, model.conv2d_layers[4], 1, 1, 0, 0, 1, 1); // 默认步长，无填充
        
        // 初始化 Inception 模块
        // InceptionA 模块
        Mixed_5b = std::make_unique<InceptionA>(
            ctx, model,
            5, 6, 7, 8, 9, 10, 11
        );
        
        Mixed_5c = std::make_unique<InceptionA>(
            ctx, model,
            12, 13, 14, 15, 16, 17, 18
        );
        
        Mixed_5d = std::make_unique<InceptionA>(
            ctx, model,
            19, 20, 21, 22, 23, 24, 25
        );
        
        // InceptionB 模块
        Mixed_6a = std::make_unique<InceptionB>(
            ctx, model,
            26, 27, 28, 29
        );
        
        // InceptionC 模块
        Mixed_6b = std::make_unique<InceptionC>(
            ctx, model,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39
        );
        
        Mixed_6c = std::make_unique<InceptionC>(
            ctx, model,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49
        );
        
        Mixed_6d = std::make_unique<InceptionC>(
            ctx, model,
            50, 51, 52, 53, 54, 55, 56, 57, 58, 59
        );
        
        Mixed_6e = std::make_unique<InceptionC>(
            ctx, model,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 69
        );
        
        // // 辅助分类器
        // if (use_aux_logits) {
        //     AuxLogits = std::make_unique<InceptionAux>(
        //         ctx, model,
        //         70, 71, 0  // 假设辅助分类器使用的是 dense_layers[0]
        //     );
        // }
        
        // InceptionD 模块
        Mixed_7a = std::make_unique<InceptionD>(
            ctx, model,
            70, 71, 72, 73, 74, 75
        );
        
        // InceptionE 模块
        Mixed_7b = std::make_unique<InceptionE>(
            ctx, model,
            76, 77, 78, 79, 80, 81, 82, 83, 84
        );
        
        Mixed_7c = std::make_unique<InceptionE>(
            ctx, model,
            85, 86, 87, 88, 89, 90, 91, 92, 93
        );
        
        // 最终分类层
        fc = std::make_unique<BasicDense>(ctx, model.dense_layers[0]);
    }
    
    // 前向传播
    ggml_tensor* apply(ggml_tensor* input) override {
        // Stem 部分
        struct ggml_tensor* x = Conv2d_1a_3x3->apply(input);
        ggml_set_name(x, "Conv2d_1a_3x3");
        
        x = Conv2d_2a_3x3->apply(x);
        ggml_set_name(x, "Conv2d_2a_3x3");
        
        x = Conv2d_2b_3x3->apply(x);
        ggml_set_name(x, "Conv2d_2b_3x3");
        
        // 第一个最大池化
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
        ggml_set_name(x, "maxpool1");
        
        x = Conv2d_3b_1x1->apply(x);
        ggml_set_name(x, "Conv2d_3b_1x1");
        
        x = Conv2d_4a_3x3->apply(x);
        ggml_set_name(x, "Conv2d_4a_3x3");
        
        // 第二个最大池化
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
        ggml_set_name(x, "maxpool2");
        
        // Inception 模块
        x = Mixed_5b->apply(x);
        ggml_set_name(x, "Mixed_5b");
        
        x = Mixed_5c->apply(x);
        ggml_set_name(x, "Mixed_5c");
        
        x = Mixed_5d->apply(x);
        ggml_set_name(x, "Mixed_5d");
        
        x = Mixed_6a->apply(x);
        ggml_set_name(x, "Mixed_6a");
        
        x = Mixed_6b->apply(x);
        ggml_set_name(x, "Mixed_6b");
        
        x = Mixed_6c->apply(x);
        ggml_set_name(x, "Mixed_6c");
        
        x = Mixed_6d->apply(x);
        ggml_set_name(x, "Mixed_6d");
        
        x = Mixed_6e->apply(x);
        ggml_set_name(x, "Mixed_6e");
        
        // 辅助分类器输出
        // struct ggml_tensor* aux = nullptr;
        // if (use_aux_logits && AuxLogits) {
        //     aux = AuxLogits->apply(x);
        //     ggml_set_name(aux, "AuxLogits");
        // }
        
        x = Mixed_7a->apply(x);
        ggml_set_name(x, "Mixed_7a");
        
        x = Mixed_7b->apply(x);
        ggml_set_name(x, "Mixed_7b");
        
        x = Mixed_7c->apply(x);
        ggml_set_name(x, "Mixed_7c");
        
        // 全局平均池化
        x = ggml_pool_2d(
            ctx, x, GGML_OP_POOL_AVG,
            x->ne[0], x->ne[1],  // kernel size = feature map size
            1, 1,  // stride
            0, 0   // padding
        );
        ggml_set_name(x, "avgpool");
        
        // Dropout (在推理时不需要)
        
        // 分类层
        x = fc->apply(x);
        ggml_set_name(x, "fc");
        
        return x;
    }
    
    // 获取辅助分类器输出
    ggml_tensor* get_aux_logits(ggml_tensor* input) {
        if (!use_aux_logits || !AuxLogits) {
            return nullptr;
        }
        
        // 执行到辅助分类器位置
        struct ggml_tensor* x = Conv2d_1a_3x3->apply(input);
        x = Conv2d_2a_3x3->apply(x);
        x = Conv2d_2b_3x3->apply(x);
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
        x = Conv2d_3b_1x1->apply(x);
        x = Conv2d_4a_3x3->apply(x);
        x = ggml_pool_2d(ctx, x, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
        x = Mixed_5b->apply(x);
        x = Mixed_5c->apply(x);
        x = Mixed_5d->apply(x);
        x = Mixed_6a->apply(x);
        x = Mixed_6b->apply(x);
        x = Mixed_6c->apply(x);
        x = Mixed_6d->apply(x);
        x = Mixed_6e->apply(x);
        
        return AuxLogits->apply(x);
    }
};

// 构造函数
Inception3Model::Inception3Model() = default;

// 析构函数
Inception3Model::~Inception3Model() {
    if (backend) {
        ggml_backend_free(backend);
    }
    if (buffer) {
        ggml_backend_buffer_free(buffer);
    }
    if (ctx) {
        ggml_free(ctx);
    }
}

// 加载模型
bool Inception3Model::load_model(const std::string& fname) {
    // 初始化后端
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // 初始化设备 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init();
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif

    // 如果没有GPU后端，则回退到CPU后端
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    
    struct ggml_context* tmp_ctx = nullptr;
    struct gguf_init_params gguf_params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &tmp_ctx,
    };
    gguf_context* gguf_ctx = gguf_init_from_file(fname.c_str(), gguf_params);
    if (!gguf_ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    int num_tensors = gguf_get_n_tensors(gguf_ctx);

    // size_t mem_required = 1024 * 1024 * 1024; // 1GB
    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors * 16,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };
    ctx = ggml_init(params);
    for (int i = 0; i < num_tensors; i++) {
        const char* name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor* src = ggml_get_tensor(tmp_ctx, name);
        struct ggml_tensor* dst = ggml_dup_tensor(ctx, src);
        ggml_set_name(dst, name);
        // printf("tensor %d: %s\n", i, name);
    }
    buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    
    // 将张量从主内存复制到后端
    for (struct ggml_tensor* cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) {
        struct ggml_tensor* src = ggml_get_tensor(tmp_ctx, ggml_get_name(cur));
        size_t n_size = ggml_nbytes(src);
        ggml_backend_tensor_set(cur, ggml_get_data(src), 0, n_size);
    }
    gguf_free(gguf_ctx);

    // 初始化模型层
    conv2d_layers.resize(94);
    dense_layers.resize(1);
    char name[256];

    // 第0个卷积层
    {
        snprintf(name, sizeof(name), "conv2d_kernel");
        conv2d_layers[0].weights = ggml_get_tensor(ctx, name);
        GGML_ASSERT(conv2d_layers[0].weights != NULL);
        // GGML_ASSERT(conv2d_layers[0].weights->type == GGML_TYPE_F16);

        if (conv2d_layers[0].batch_normalize) {
            snprintf(name, sizeof(name), "batch_normalization_beta");
            conv2d_layers[0].beta = ggml_get_tensor(ctx, name);
            GGML_ASSERT(conv2d_layers[0].beta != NULL);
            GGML_ASSERT(conv2d_layers[0].beta->type == GGML_TYPE_F32);
            
            snprintf(name, sizeof(name), "batch_normalization_moving_mean");
            conv2d_layers[0].moving_mean = ggml_get_tensor(ctx, name);
            GGML_ASSERT(conv2d_layers[0].moving_mean != NULL);
            GGML_ASSERT(conv2d_layers[0].moving_mean->type == GGML_TYPE_F32);
            
            snprintf(name, sizeof(name), "batch_normalization_moving_variance");
            conv2d_layers[0].moving_variance = ggml_get_tensor(ctx, name);
            GGML_ASSERT(conv2d_layers[0].moving_variance != NULL);
            GGML_ASSERT(conv2d_layers[0].moving_variance->type == GGML_TYPE_F32);
        }
    }
    
    // 其他卷积层
    for (int i = 1; i < (int)conv2d_layers.size(); i++) {
        snprintf(name, sizeof(name), "conv2d_%d_kernel", i);
        conv2d_layers[i].weights = ggml_get_tensor(ctx, name);
        GGML_ASSERT(conv2d_layers[i].weights != NULL);
        // GGML_ASSERT(conv2d_layers[i].weights->type == GGML_TYPE_F16);
        
        if (conv2d_layers[i].batch_normalize) {
            snprintf(name, sizeof(name), "batch_normalization_%d_beta", i);
            conv2d_layers[i].beta = ggml_get_tensor(ctx, name);
            GGML_ASSERT(conv2d_layers[i].beta != NULL);
            GGML_ASSERT(conv2d_layers[i].beta->type == GGML_TYPE_F32);
            
            snprintf(name, sizeof(name), "batch_normalization_%d_moving_mean", i);
            conv2d_layers[i].moving_mean = ggml_get_tensor(ctx, name);
            GGML_ASSERT(conv2d_layers[i].moving_mean != NULL);
            GGML_ASSERT(conv2d_layers[i].moving_mean->type == GGML_TYPE_F32);
            
            snprintf(name, sizeof(name), "batch_normalization_%d_moving_variance", i);
            conv2d_layers[i].moving_variance = ggml_get_tensor(ctx, name);
            GGML_ASSERT(conv2d_layers[i].moving_variance != NULL);
            GGML_ASSERT(conv2d_layers[i].moving_variance->type == GGML_TYPE_F32);
        }
    }
    
    // 分类层（Dense）
    {
        snprintf(name, sizeof(name), "classification_kernel");
        dense_layers[0].kernel = ggml_get_tensor(ctx, name);
        GGML_ASSERT(dense_layers[0].kernel != NULL);
        // GGML_ASSERT(dense_layers[0].kernel->type == GGML_TYPE_F16);
        
        snprintf(name, sizeof(name), "classification_bias");
        dense_layers[0].biases = ggml_get_tensor(ctx, name);
        GGML_ASSERT(dense_layers[0].biases != NULL);
        GGML_ASSERT(dense_layers[0].biases->type == GGML_TYPE_F32);
    }

    return true;
}

// 构建计算图
struct ggml_cgraph* Inception3Model::build_graph(int batch_size) {
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    bool use_aux_logits = false;
    
    // 创建输入张量
    images = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, width, height, channels, batch_size);
    ggml_set_name(images, "images");
    ggml_set_input(images);
    
    // 创建 Inception3 网络
    std::unique_ptr<Inception3> inception3 = std::make_unique<Inception3>(ctx, *this, use_aux_logits);
    
    // 应用网络
    classification = inception3->apply(images);
    ggml_set_name(classification, "classification");
    
    // 构建图
    ggml_build_forward_expand(gf, classification);
    
    return gf;
}

// 用于调试的函数，打印指定名称的中间张量
void Inception3Model::print_intermediate_tensor(const char* name) {
    ggml_tensor* tensor = find_tensor_by_name(ctx, name);
    if (tensor) {
        print_tensor_info(name, tensor);
    } else {
        fprintf(stderr, "Tensor with name '%s' not found\n", name);
    }
}

// 执行推理并打印结果
bool Inception3Model::infer(struct ggml_cgraph * gf) {
    if (!ctx || !backend || !buffer) {
        fprintf(stderr, "Model not properly initialized\n");
        return false;
    }
    
    // 执行计算
    printf("开始计算...\n");
    const int64_t t_start_ms = ggml_time_ms();
    
    // 执行计算
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "%s: ggml_backend_graph_compute() 失败\n", __func__);
        return false;}
    
    const int64_t t_end_ms = ggml_time_ms();
    printf("计算完成，用时: %lld ms\n", t_end_ms - t_start_ms);

    // 打印分类结果
    fprintf(stderr, "\n=== Classification Result ===\n\n");
    print_tensor_info("classification", classification);
    
    return true;
}


// 测试指定层的函数
bool test_inception3_layer(const std::string& model_path, const std::string& layer_name) {
    printf("正在测试 %s 层...\n", layer_name.c_str());
    
    // 初始化模型
    Inception3Model model;
    if (!model.load_model(model_path)) {
        fprintf(stderr, "无法加载模型: %s\n", model_path.c_str());
        return false;
    }
    
    // 创建 GGML 计算用的上下文
    struct ggml_init_params params0 = {
        /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
    };
    struct ggml_context * ctx_cgraph = ggml_init(params0);
    
    // 定义需要依次构建的层
    std::vector<std::string> layer_sequence = {
        "Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1",
        "Conv2d_3b_1x1", "Conv2d_4a_3x3", "maxpool2",
        "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a",
        "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e",
        "Mixed_7a", "Mixed_7b", "Mixed_7c", "avgpool", "fc"
    };
    
    // 查找目标层在序列中的位置
    int target_layer_idx = -1;
    for (size_t i = 0; i < layer_sequence.size(); ++i) {
        if (layer_sequence[i] == layer_name) {
            target_layer_idx = i;
            break;
        }
    }
    
    if (target_layer_idx == -1) {
        fprintf(stderr, "未找到层: %s\n", layer_name.c_str());
        ggml_free(ctx_cgraph);
        return false;
    }
    
    // 创建 Inception3 网络
    std::unique_ptr<Inception3> inception3 = std::make_unique<Inception3>(ctx_cgraph, model, false);
    
    // 创建输入张量
    struct ggml_tensor* input = ggml_new_tensor_4d(ctx_cgraph, GGML_TYPE_F32, model.width, model.height, model.channels, 1);
    ggml_set_name(input, "input");
    ggml_set_input(input);
    
    // 构建当前层的计算图
    struct ggml_cgraph* gf = ggml_new_graph(ctx_cgraph);
    if (!gf) {
        fprintf(stderr, "构建计算图失败\n");
        ggml_free(ctx_cgraph);
        return false;
    }
    
    // 根据目标层构建计算图
    struct ggml_tensor* result = nullptr;
    
    // 应用网络直到目标层
    // Stem 部分
    if (target_layer_idx >= 0) { // Conv2d_1a_3x3
        result = inception3->Conv2d_1a_3x3->apply(input);
        ggml_set_name(result, "Conv2d_1a_3x3");
    }
    
    if (target_layer_idx >= 1) { // Conv2d_2a_3x3
        result = inception3->Conv2d_2a_3x3->apply(result);
        ggml_set_name(result, "Conv2d_2a_3x3");
    }
    
    if (target_layer_idx >= 2) { // Conv2d_2b_3x3
        result = inception3->Conv2d_2b_3x3->apply(result);
        ggml_set_name(result, "Conv2d_2b_3x3");
    }
    
    if (target_layer_idx >= 3) { // maxpool1
        result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
        ggml_set_name(result, "maxpool1");
    }
    
    if (target_layer_idx >= 4) { // Conv2d_3b_1x1
        result = inception3->Conv2d_3b_1x1->apply(result);
        ggml_set_name(result, "Conv2d_3b_1x1");
    }
    
    if (target_layer_idx >= 5) { // Conv2d_4a_3x3
        result = inception3->Conv2d_4a_3x3->apply(result);
        ggml_set_name(result, "Conv2d_4a_3x3");
    }
    
    if (target_layer_idx >= 6) { // maxpool2
        result = ggml_pool_2d(ctx_cgraph, result, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
        ggml_set_name(result, "maxpool2");
    }
    
    // Inception 模块
    if (target_layer_idx >= 7) { // Mixed_5b
        result = inception3->Mixed_5b->apply(result);
        ggml_set_name(result, "Mixed_5b");
    }
    
    if (target_layer_idx >= 8) { // Mixed_5c
        result = inception3->Mixed_5c->apply(result);
        ggml_set_name(result, "Mixed_5c");
    }
    
    if (target_layer_idx >= 9) { // Mixed_5d
        result = inception3->Mixed_5d->apply(result);
        ggml_set_name(result, "Mixed_5d");
    }
    
    if (target_layer_idx >= 10) { // Mixed_6a
        result = inception3->Mixed_6a->apply(result);
        ggml_set_name(result, "Mixed_6a");
    }
    
    if (target_layer_idx >= 11) { // Mixed_6b
        result = inception3->Mixed_6b->apply(result);
        ggml_set_name(result, "Mixed_6b");
    }
    
    if (target_layer_idx >= 12) { // Mixed_6c
        result = inception3->Mixed_6c->apply(result);
        ggml_set_name(result, "Mixed_6c");
    }
    
    if (target_layer_idx >= 13) { // Mixed_6d
        result = inception3->Mixed_6d->apply(result);
        ggml_set_name(result, "Mixed_6d");
    }
    
    if (target_layer_idx >= 14) { // Mixed_6e
        result = inception3->Mixed_6e->apply(result);
        ggml_set_name(result, "Mixed_6e");
    }
    
    if (target_layer_idx >= 15) { // Mixed_7a
        result = inception3->Mixed_7a->apply(result);
        ggml_set_name(result, "Mixed_7a");
    }
    
    if (target_layer_idx >= 16) { // Mixed_7b
        result = inception3->Mixed_7b->apply(result);
        ggml_set_name(result, "Mixed_7b");
    }
    
    if (target_layer_idx >= 17) { // Mixed_7c
        result = inception3->Mixed_7c->apply(result);
        ggml_set_name(result, "Mixed_7c");
    }
    
    if (target_layer_idx >= 18) { // avgpool
        result = ggml_pool_2d(
            ctx_cgraph, result, GGML_OP_POOL_AVG,
            result->ne[0], result->ne[1],  // kernel size = feature map size
            1, 1,  // stride
            0, 0   // padding
        );
        ggml_set_name(result, "avgpool");
    }
    
    if (target_layer_idx >= 19) { // fc
        result = inception3->fc->apply(result);
        ggml_set_name(result, "fc");
    }
    
    // 构建前向计算图
    if (result) {
        ggml_build_forward_expand(gf, result);
    } else {
        fprintf(stderr, "未能构建到层: %s\n", layer_name.c_str());
        ggml_free(ctx_cgraph);
        return false;
    }
    
    // 分配计算资源
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);
    
    // 设置输入张量，使用随机数据
    float* data0 = new float[model.width * model.height * model.channels];
    for (int i = 0; i < model.width * model.height * model.channels; i++) {
        data0[i] = 0.5f;
    }
    ggml_backend_tensor_set(input, data0, 0, ggml_nbytes(input));
    delete[] data0;
    
    // 执行计算
    printf("开始计算到 %s 层...\n", layer_name.c_str());
    if (ggml_backend_graph_compute(model.backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "执行计算失败\n");
        ggml_free(ctx_cgraph);
        ggml_gallocr_free(allocr);
        return false;
    }
    
    // 打印目标层的输出
    printf("\n=== %s 层的输出 ===\n", layer_name.c_str());
    print_tensor_info(layer_name.c_str(), result);
    
    // 释放资源
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    
    printf("\n%s 层测试完成！\n", layer_name.c_str());
    return true;
}