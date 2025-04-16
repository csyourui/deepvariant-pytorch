#!/usr/bin/env python3
import argparse
import os

import gguf
import numpy as np
import tensorflow as tf

# 导入 inceptionv3 定义
from tensorflow_model.keras_modeling import inceptionv3 as tf_inception_v3

NUM_CLASSES = 3
INPUT_SHAPE = (100, 221, 7)


def load_tiny_cnn_model(weights_path):
    """
    加载 tiny_cnn 模型。

    Args:
        weights_path: 模型权重路径 (.h5 文件)

    Returns:
        加载的 TensorFlow 模型
    """
    print(f"Loading TensorFlow model from {weights_path}...")
    model = tf.keras.models.load_model(weights_path)
    model.summary()
    return model


def save_layer_params(gguf_writer, tf_layer, prefix):
    """
    将 TensorFlow 层的参数保存到 GGUF 格式。

    Args:
        gguf_writer: GGUF 写入器实例
        tf_layer: TensorFlow 层对象
        prefix: 参数名称前缀
    """

    for weight in tf_layer.weights:
        # 获取权重名称和值
        layer_name = tf_layer.name
        weight_name = weight.name.split("/")[-1].split(":")[0]
        weight_value = weight.numpy()
        print(
            f"\t tf: Layer: {layer_name}, Weight: {weight_name}, Shape: {weight_value.shape}, size: {weight_value.size}"
        )

        # 根据权重类型进行特殊处理
        if "kernel" in weight_name and "conv2d" in layer_name:
            # 卷积层权重需要转置为 GGUF 格式
            if len(weight_value.shape) == 4:
                # TensorFlow: [H, W, input_channels, output_channels]
                # GGUF: [output_channels, input_channels, H, W]
                weight_value = np.transpose(weight_value, (3, 2, 0, 1))

            # 将权重转换为 float16 以节省空间
            weight_value = weight_value.astype(np.float16)

        if "batch_normalization" in layer_name:
            # 批归一化层权重需要转置为 GGUF 格式
            if len(weight_value.shape) == 1:
                weight_value = weight_value.reshape((1, -1, 1, 1))  # [1, 1, C, N]
            # 如果是 moving_variance，则需要加上一个小的常数eps以避免除零错误
            if "moving_variance" in weight_name:
                weight_value = weight_value + tf_layer.epsilon
            weight_value = weight_value.astype(np.float32)

        if "kernel" in weight_name and "classification" in layer_name:
            # 全连接层权重需要转置为 GGUF 格式
            if len(weight_value.shape) == 2:
                # TensorFlow: [input_size, output_size]
                # GGUF: [output_size, input_size]
                weight_value = weight_value.T
                weight_value = weight_value.reshape(
                    (1, 1, weight_value.shape[0], weight_value.shape[1])
                )
            # 将权重转换为 float16 以节省空间
            weight_value = weight_value.astype(np.float16)

        if "bias" in weight_name and "classification" in layer_name:
            if len(weight_value.shape) == 1:
                weight_value = weight_value.reshape((1, 1, 1, weight_value.shape[0]))
            # 将偏置转换为 float16
            weight_value = weight_value.astype(np.float32)

        # 构造 GGUF 参数名称
        gguf_name = f"{prefix}_{weight_name}"

        # 添加张量到 GGUF 文件
        gguf_writer.add_tensor(gguf_name, weight_value, raw_shape=weight_value.shape)

        print(f"\t gg: Added tensor: {gguf_name} with shape {weight_value.shape}")


def tf2gguf(tf_model, output_path):
    """
    将 TensorFlow inceptionv3 模型转换为 GGUF 格式。

    Args:
        tf_model: TensorFlow inceptionv3 模型
        output_path: 输出 GGUF 文件路径
    """
    # 创建 GGUF 写入器
    model_name = os.path.basename(output_path).split(".")[0]
    gguf_writer = gguf.GGUFWriter(output_path, model_name)

    # 遍历模型的所有层
    for i, layer in enumerate(tf_model.layers):
        print(
            "-------------------------------------------------------------------------"
        )
        # 跳过非参数化层
        if not layer.weights:
            print(f"Skipping layer: {layer.name}")
            continue

        print(
            f"Processing layer: {layer.name}, Type: {layer.__class__.__name__}, weight blocks: {len(layer.weights)}"
        )

        # 保存层参数
        save_layer_params(gguf_writer, layer, layer.name)

    # 写入 GGUF 文件
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print("-------------------------------------------------------------------------")
    print(f"模型已成功转换为 GGUF 格式: {output_path}")
    return output_path


def test_model_conversion(tf_model, gguf_path):
    """
    测试模型转换是否成功，使用相同的输入数据比较输出。

    Args:
        tf_model: 原始 TensorFlow 模型
        gguf_path: 转换后的 GGUF 模型路径
    """
    print("\n==== 验证模型转换 ====")

    # 验证 GGUF 文件是否存在
    if os.path.exists(gguf_path):
        file_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        print(f"GGUF 模型文件大小: {file_size_mb:.2f} MB")
        print("GGUF 模型文件已成功创建！")
    else:
        print("错误：GGUF 模型文件未创建！")
        return

    display_gguf_info(gguf_path)
    # 创建随机测试输入
    np.random.seed(42)  # 设置随机种子以保证结果可重现
    test_input = np.random.random((1, 100, 221, 7)).astype(np.float32)
    test_input = 0.1 * np.ones((1, 100, 221, 7), dtype=np.float32)
    print(f"测试输入形状: {test_input.shape}")

    # 获取 TensorFlow 模型预测结果
    tf_output = tf_model.predict(test_input)
    print(f"TensorFlow 模型输出形状: {tf_output.shape}")
    print(f"TensorFlow 预测概率: {tf_output[0]}")
    print(f"预测类别: {np.argmax(tf_output[0])}")


def display_gguf_info(gguf_path):
    """显示 GGUF 模型的基本信息"""
    # 读取 GGUF 文件
    reader = gguf.GGUFReader(gguf_path)

    # 显示基本信息
    # print(f"模型名称: {reader.name}")
    # print(f"模型架构: {reader.arch}")
    print(f"张量数量: {len(reader.tensors)}")

    # 列出所有张量
    print("\n张量列表:")
    for i, _ in enumerate(reader.tensors):
        tensor = reader.get_tensor(i)
        print(f"{i}. {tensor.name}: {tensor.shape}:{tensor.tensor_type}")

    # 列出元数据
    print("\n模型元数据:")
    for key in reader.fields:
        print(f"{key}: {reader.fields[key]}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Convert tiny_cnn model to GGUF format"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="data/tf_model/deepvariant.wgs.h5",
        help="Path to existing model weights (.h5 file)",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate and train a new model if weights not provided",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gguf_model/deepvariant.gguf",
        help="Output GGUF file path",
    )

    args = parser.parse_args()

    tf_model = tf_inception_v3(weights=args.weights)
    # 转换为 GGUF 格式
    gguf_path = tf2gguf(tf_model, args.output)

    # 测试模型转换
    test_model_conversion(tf_model, gguf_path)
    print("\n转换和测试完成！")


if __name__ == "__main__":
    main()
