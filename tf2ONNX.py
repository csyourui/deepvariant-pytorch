#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert TensorFlow inception model to ONNX format
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
    Convert TensorFlow model to ONNX format with support for dynamic batch_size

    Args:
        tf_model: TensorFlow model file
        onnx_model_path: Output ONNX model file path
    """

    # Output model summary
    tf_model.summary()

    # Get model input shape
    input_shape = tf_model.input_shape
    print(f"Original model input shape: {input_shape}")

    # Keep batch size as None, but use batch_size=1 for creating example input
    concrete_input_shape = list(input_shape)
    if concrete_input_shape[0] is None:
        concrete_input_shape[0] = 1

    print(f"Model input shape used for testing: {concrete_input_shape}")

    # Create sample input data (for testing only)
    if (
        len(concrete_input_shape) == 4
    ):  # Image input (batch_size, height, width, channels)
        dummy_input = np.random.random(concrete_input_shape).astype(np.float32)
    else:
        print(
            f"Warning: Unusual input shape: {concrete_input_shape}, please adjust sample input manually"
        )
        dummy_input = np.random.random(concrete_input_shape).astype(np.float32)

    # Convert to ONNX
    print("Converting model to ONNX format...")

    # Get dynamic input shape (None replaced by "batch")
    dynamic_input_shape = list(input_shape)
    if dynamic_input_shape[0] is None:
        dynamic_input_shape[0] = "batch"

    # Specify input signature with dynamic batch size
    input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")]

    # Define dynamic axes mapping, specify first dimension as dynamic
    dynamic_axes = {"input": {0: "batch"}}
    for i, output in enumerate(tf_model.outputs):
        output_name = (
            f"output_{i}" if not hasattr(output, "name") else output.name.split(":")[0]
        )
        dynamic_axes[output_name] = {0: "batch"}

    # Convert model with explicit dynamic axes
    onnx_model, _ = tf2onnx.convert.from_keras(
        tf_model,
        input_signature=input_signature,
        opset=13,
        output_path=onnx_model_path,
    )

    print(f"ONNX model saved to: {onnx_model_path}")

    # Validate ONNX model
    try:
        import onnxruntime as ort

        print("Validating ONNX model...")

        # Use ONNX Runtime for inference
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess = ort.InferenceSession(onnx_model_path, sess_options)

        input_name = sess.get_inputs()[0].name

        # Validate single example
        result = sess.run(None, {input_name: dummy_input})
        tf_result = tf_model.predict(dummy_input)

        # Compare results
        print(
            f"TensorFlow output shape: {tf_result.shape}, ONNX output shape: {result[0].shape}"
        )

        # Output sample results for visual comparison
        print(f"TensorFlow sample output: {tf_result[0, :3]}")
        print(f"ONNX sample output: {result[0][0, :3]}")

        # Compare results with tolerance for small errors
        np.testing.assert_allclose(result[0], tf_result, rtol=1e-5, atol=1e-5)
        print("Single sample validation successful!")

        # Additional test with different batch_size
        batch_size = 3
        larger_input = np.random.random(
            (batch_size,) + tuple(concrete_input_shape[1:])
        ).astype(np.float32)
        larger_result = sess.run(None, {input_name: larger_input})
        print(
            f"Validation passed with batch_size={batch_size}, output shape: {larger_result[0].shape}"
        )

        print("Validation successful! Model supports dynamic batch_size.")
    except ImportError:
        print("Warning: onnxruntime not installed, skipping model validation step.")
    except Exception as e:
        print(f"Model validation failed: {e}")

    print("Conversion complete!")
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
        # required=True,
        default="data/tf_model/deepvariant.wgs.ckpt",
        help="输入的TensorFlow模型文件路径 (.h5或检查点文件)",
    )
    parser.add_argument(
        "--output",
        type=str,
        # required=True,
        default="data/onnx_model/deepvariant.onnx",
        help="输出的ONNX模型文件路径",
    )
    args = parser.parse_args()

    # 确保文件路径是绝对路径
    input_path = os.path.abspath(args.weights)
    output_path = os.path.abspath(args.output)

    # 检查输入文件是否存在
    if not os.path.exists(input_path) and not os.path.exists(input_path + ".index"):
        print(f"Error: Input file not found {input_path}")
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
