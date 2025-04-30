#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将TensorFlow的inception模型转换为ONNX格式
"""

import argparse
import os

import numpy as np
import tensorflow as tf

import tf2onnx

# 导入inceptionv3定义
from tensorflow_model.keras_modeling import inceptionv3 as tf_inception_v3

NUM_CLASSES = 3
INPUT_SHAPE = (100, 221, 7)


def convert_tf_to_onnx(tf_model, onnx_model_path):
    """
    将TensorFlow格式的模型转换为ONNX格式

    Args:
        tf_model: TensorFlow模型文件
        onnx_model_path: 输出的ONNX模型文件路径
    """

    # 输出模型摘要信息
    tf_model.summary()

    # 获取模型输入形状并处理None批次大小
    input_shape = tf_model.input_shape
    print(f"原始模型输入形状: {input_shape}")

    # 将None替换为1（批次大小）
    concrete_input_shape = list(input_shape)
    if concrete_input_shape[0] is None:
        concrete_input_shape[0] = 1

    print(f"使用的模型输入形状: {concrete_input_shape}")

    # 创建一个示例输入数据
    if len(concrete_input_shape) == 4:  # 图像输入 (batch_size, height, width, channels)
        dummy_input = np.random.random(concrete_input_shape).astype(np.float32)
    else:
        print(f"警告: 不常见的输入形状: {concrete_input_shape}，请手动调整示例输入")
        dummy_input = np.random.random(concrete_input_shape).astype(np.float32)

    # 转换为ONNX
    print("转换模型为ONNX格式...")

    # 指定输入和输出名称，使用具体的batch大小
    input_signature = [tf.TensorSpec(concrete_input_shape, tf.float32, name="input")]

    # 转换模型，指定动态轴（第一个维度）
    onnx_model, _ = tf2onnx.convert.from_keras(
        tf_model, input_signature=input_signature, opset=13, output_path=onnx_model_path
    )

    print(f"ONNX模型已保存至: {onnx_model_path}")

    # 验证ONNX模型
    try:
        import onnxruntime as ort

        print("验证ONNX模型...")

        # 使用ONNX Runtime进行推理
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess = ort.InferenceSession(onnx_model_path, sess_options)

        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: dummy_input})

        # 使用原始TensorFlow模型进行推理
        tf_result = tf_model.predict(dummy_input)

        # 比较结果
        print(f"TensorFlow输出形状: {tf_result.shape}, ONNX输出形状: {result[0].shape}")

        # 输出结果样本进行直观比较
        print(f"TensorFlow样本输出: {tf_result[0, :3]}")
        print(f"ONNX样本输出: {result[0][0, :3]}")

        # 进行结果比较，容忍小误差
        np.testing.assert_allclose(result[0], tf_result, rtol=1e-5, atol=1e-5)
        print("验证成功! TensorFlow和ONNX模型输出一致。")
    except ImportError:
        print("警告: 未安装onnxruntime，跳过模型验证步骤。")
    except Exception as e:
        print(f"模型验证失败: {e}")

    print("转换完成!")
    return onnx_model_path


def load_inception_model(weights_path):
    """
    加载inception模型

    Args:
        weights_path: 模型权重路径

    Returns:
        加载的TensorFlow模型
    """
    try:
        if weights_path.endswith(".h5"):
            # 直接加载.h5文件
            model = tf.keras.models.load_model(weights_path)
        else:
            # 使用项目中定义的函数加载
            model = tf_inception_v3(weights=weights_path)

        return model
    except Exception as e:
        print(f"加载模型失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="将TensorFlow inception模型转换为ONNX格式"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="输入的TensorFlow模型文件路径 (.h5或检查点文件)",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="输出的ONNX模型文件路径"
    )
    args = parser.parse_args()

    # 确保文件路径是绝对路径
    input_path = os.path.abspath(args.weights)
    output_path = os.path.abspath(args.output)

    # 检查输入文件是否存在
    if not os.path.exists(input_path) and not os.path.exists(input_path + ".index"):
        print(f"错误: 找不到输入文件 {input_path}")
        return 1

    try:
        # 加载模型
        tf_model = load_inception_model(input_path)

        # 使用加载的模型直接转换为ONNX
        convert_tf_to_onnx(tf_model, output_path)
        return 0
    except Exception as e:
        print(f"转换失败: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
