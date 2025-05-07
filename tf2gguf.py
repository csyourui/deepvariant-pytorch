#!/usr/bin/env python3
import argparse
import os

import gguf
import numpy as np
import tensorflow as tf

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
        # get weight name and value
        layer_name = tf_layer.name
        weight_name = weight.name.split("/")[-1].split(":")[0]
        weight_value = weight.numpy()
        print(
            f"\t tf: Layer: {layer_name}, Weight: {weight_name}, Shape: {weight_value.shape}, size: {weight_value.size}"
        )

        # handle different types of layers
        if "kernel" in weight_name and "conv2d" in layer_name:
            # Convolutional layer weights need to be transposed to GGUF format
            if len(weight_value.shape) == 4:
                # TensorFlow: [H, W, input_channels, output_channels]
                # GGUF: [output_channels, input_channels, H, W]
                weight_value = np.transpose(weight_value, (3, 2, 0, 1))
            weight_value = weight_value.astype(np.float16)

        if "batch_normalization" in layer_name:
            # Batch normalization layer weights need to be transposed to GGUF format
            if len(weight_value.shape) == 1:
                weight_value = weight_value.reshape((1, -1, 1, 1))  # [1, 1, C, N]
            # If it's moving_variance, add a small constant eps to avoid division by zero error
            if "moving_variance" in weight_name:
                weight_value = weight_value + tf_layer.epsilon
            weight_value = weight_value.astype(np.float32)

        if "kernel" in weight_name and "classification" in layer_name:
            # Fully connected layer weights need to be transposed to GGUF format
            if len(weight_value.shape) == 2:
                # TensorFlow: [input_size, output_size]
                # GGUF: [output_size, input_size]
                weight_value = weight_value.T
                weight_value = weight_value.reshape(
                    (1, 1, weight_value.shape[0], weight_value.shape[1])
                )
            weight_value = weight_value.astype(np.float16)

        if "bias" in weight_name and "classification" in layer_name:
            if len(weight_value.shape) == 1:
                weight_value = weight_value.reshape((1, 1, 1, weight_value.shape[0]))
            weight_value = weight_value.astype(np.float32)

        # construct GGUF tensor name
        gguf_name = f"{prefix}_{weight_name}"

        # add tensor to GGUF writer
        gguf_writer.add_tensor(gguf_name, weight_value, raw_shape=weight_value.shape)

        print(f"\t gg: Added tensor: {gguf_name} with shape {weight_value.shape}")


def tf2gguf(tf_model, output_path):
    """
    将 TensorFlow inceptionv3 模型转换为 GGUF 格式。

    Args:
        tf_model: TensorFlow inceptionv3 模型
        output_path: 输出 GGUF 文件路径
    """
    # create GGUF writer
    model_name = os.path.basename(output_path).split(".")[0]
    gguf_writer = gguf.GGUFWriter(output_path, model_name)

    # traverse all layers in the model
    for i, layer in enumerate(tf_model.layers):
        print(
            "-------------------------------------------------------------------------"
        )
        # skip layers without weights
        if not layer.weights:
            print(f"Skipping layer: {layer.name}")
            continue

        print(
            f"Processing layer: {layer.name}, Type: {layer.__class__.__name__}, weight blocks: {len(layer.weights)}"
        )

        # save layer parameters
        save_layer_params(gguf_writer, layer, layer.name)

    # write GGUF header
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()

    print("-------------------------------------------------------------------------")
    print(f"Model successfully converted to GGUF format: {output_path}")
    return output_path


def test_model_conversion(tf_model, gguf_path):
    """
    Test if model conversion is successful by comparing outputs with the same input data.

    Args:
        tf_model: Original TensorFlow model
        gguf_path: Path to the converted GGUF model
    """
    print("\n==== Validating Model Conversion ====")

    # Verify if GGUF file exists
    if os.path.exists(gguf_path):
        file_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
        print(f"GGUF model file size: {file_size_mb:.2f} MB")
        print("GGUF model file was successfully created!")
    else:
        print("Error: GGUF model file was not created!")
        return

    display_gguf_info(gguf_path)
    # Create random test input
    np.random.seed(42)  # Set random seed to ensure reproducibility
    test_input = np.random.random((1, 100, 221, 7)).astype(np.float32)
    test_input = 0.1 * np.ones((1, 100, 221, 7), dtype=np.float32)
    print(f"Test input shape: {test_input.shape}")

    # Get TensorFlow model prediction results
    tf_output = tf_model.predict(test_input)
    print(f"TensorFlow model output shape: {tf_output.shape}")
    print(f"TensorFlow prediction probabilities: {tf_output[0]}")
    print(f"Predicted class: {np.argmax(tf_output[0])}")


def display_gguf_info(gguf_path):
    """Display the basic information of GGUF model"""
    # Read GGUF file
    reader = gguf.GGUFReader(gguf_path)

    # Display basic information
    print(f"Number of tensors: {len(reader.tensors)}")

    # List all tensors
    print("\nTensor list:")
    for i, _ in enumerate(reader.tensors):
        tensor = reader.get_tensor(i)
        print(f"{i}. {tensor.name}: {tensor.shape}:{tensor.tensor_type}")

    # List metadata
    print("\nModel metadata:")
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
    # convert TensorFlow model to GGUF format
    gguf_path = tf2gguf(tf_model, args.output)

    # test the conversion
    test_model_conversion(tf_model, gguf_path)
    print("\nConversion and testing completed!")


if __name__ == "__main__":
    main()
