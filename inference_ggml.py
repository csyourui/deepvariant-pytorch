import argparse
import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tfrecord.torch.dataset import TFRecordDataset

from ggml_model.inceptionv3_ggml import InceptionV3GGML

INPUT_SHAPE = (100, 221, 7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_with_tfrecord(data_path):
    # 定义 TFRecord 的解析规则
    description = {
        "image/encoded": "byte",
        "label": "int",
    }

    # 使用 TFRecordDataset 加载数据
    dataset = TFRecordDataset(
        data_path,
        compression_type="gzip",
        index_path=None,
        description=description,
    )

    images = []
    labels = []

    for record in dataset:
        # 解码图像
        image = np.frombuffer(record["image/encoded"], dtype=np.uint8).reshape(
            INPUT_SHAPE
        )
        # 转换维度顺序: (height, width, channels) -> (channels, height, width)
        image = np.transpose(image, (2, 0, 1))
        # 归一化
        image = (image - 128.0) / 128.0
        # 获取标签
        label = np.frombuffer(record["label"], dtype=np.int64)
        label = label.reshape(-1)

        images.append(image)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    logger.info(f"Loaded {len(images)} images with shape {images.shape}")
    logger.info(f"Loaded {len(labels)} labels with shape {labels.shape}")

    return images, labels


def load_model(model_path):
    """
    加载GGML模型

    Args:
        model_path: GGUF模型文件路径

    Returns:
        加载的模型
    """
    logger.info(f"加载模型: {model_path}")
    try:
        model = InceptionV3GGML(model_path)
        return model
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def run_inference(model, image_array):
    """
    使用GGML模型进行推理

    Args:
        model: 加载的InceptionV3GGML模型
        image_array: 预处理后的图像数据

    Returns:
        推理结果
    """
    logger.info(f"Starting inference on {image_array.shape[0]} samples")

    # 进行推理
    start_time = datetime.now()
    try:
        outputs = model(image_array)
        outputs = softmax(outputs, axis=1)
        end_time = datetime.now()

        # 计算推理时间
        inference_time = (end_time - start_time).total_seconds() * 1000  # 转换为毫秒

        logger.info(f"Inference completed in {inference_time:.2f} ms")
        logger.info(f"Predictions shape: {outputs.shape}")
        return outputs, inference_time
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Using GGML model for inference")
    parser.add_argument(
        "--model",
        type=str,
        default="./data/gguf_model/deepvariant.gguf",
        help="GGUF model path",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test/validation_set.with_label.tfrecord-00000-of-00024.gz",
        help="Test data",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to run inference on",
    )

    args = parser.parse_args()

    # 检查文件路径是否存在
    if not os.path.exists(args.model):
        logger.error(f"Model file does not exist: {args.model}")
        return

    if not os.path.exists(args.test_data):
        logger.error(f"Test data file does not exist: {args.test_data}")
        return

    try:
        # 加载模型
        model = load_model(args.model)

        # 加载测试数据
        images, labels = load_data_with_tfrecord(args.test_data)

        # 限制样本数量
        if args.num_samples > 0 and args.num_samples < len(images):
            images = images[: args.num_samples]
            labels = labels[: args.num_samples]

        logger.info(f"Running inference on {len(images)} samples")

        # 进行推理
        outputs, inference_time = run_inference(model, images)

        # 输出推理结果
        logger.info(f"推理完成，总耗时: {inference_time:.2f} 毫秒")
        logger.info(f"平均每个样本耗时: {inference_time / len(images):.2f} 毫秒")
        logger.info(
            f"预测输出形状: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}"
        )

        # 计算混淆矩阵和分类报告
        predictions = np.argmax(outputs, axis=1)
        m = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=[0, 1, 2])
        report = classification_report(labels, predictions, digits=4)
        logger.info("\nGGML Classification Report:\n%s", report)
        disp.plot()
        plt.title("GGML Confusion Matrix")
        plt.savefig("./data/ggml_confusion_matrix.png", dpi=300)

    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
