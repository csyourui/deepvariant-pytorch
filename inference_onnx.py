import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tfrecord.torch.dataset import TFRecordDataset

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


def load_onnx_model(model_path):
    """
    加载ONNX模型

    Args:
        model_path: ONNX模型文件路径

    Returns:
        加载的ONNX模型会话
    """
    logger.info(f"加载ONNX模型: {model_path}")
    try:
        # 设置ONNX Runtime会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # 创建推理会话
        session = ort.InferenceSession(model_path, sess_options)
        logger.info("ONNX模型加载成功")

        # 输出模型输入和输出信息
        input_details = session.get_inputs()
        output_details = session.get_outputs()
        logger.info(f"模型输入: {[input.name for input in input_details]}")
        logger.info(f"模型输入形状: {[input.shape for input in input_details]}")
        logger.info(f"模型输出: {[output.name for output in output_details]}")

        return session
    except Exception as e:
        logger.error(f"加载ONNX模型失败: {e}")
        raise


def prepare_batch_input(images, batch_size=32):
    """
    准备批量输入数据，调整为ONNX模型期望的格式

    Args:
        images: 图像数据数组
        batch_size: 批处理大小

    Returns:
        调整后的数据批次列表
    """
    num_samples = len(images)
    batches = []

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_images = images[i:end]

        # 调整输入维度顺序以匹配模型期望 [batch_size, height, width, channels]
        # 但ONNX模型期望单个样本: [1, height, width, channels]
        adjusted_batch = []
        for img in batch_images:
            # 将单个样本添加为独立的批次
            # 注意：不改变维度顺序，因为ONNX模型期望 [1, height, width, channels]，与原始TF模型一致
            adjusted_batch.append(np.expand_dims(img, axis=0))

        batches.append(adjusted_batch)

    return batches


def run_inference_onnx(session, image_array):
    """
    使用ONNX模型进行推理

    Args:
        session: ONNX Runtime会话
        image_array: 预处理后的图像数据 (N, H, W, C)

    Returns:
        推理结果和推理时间
    """
    logger.info(f"开始对 {len(image_array)} 个样本进行推理")

    # 获取模型输入名称
    input_name = session.get_inputs()[0].name

    # 记录开始时间
    start_time = time.time()

    try:
        # 为每个样本单独进行推理，然后合并结果
        all_results = []

        # 准备批次数据
        batches = prepare_batch_input(image_array, batch_size=32)

        for batch in batches:
            batch_results = []
            for single_sample in batch:
                # 运行推理，每次一个样本
                outputs = session.run(None, {input_name: single_sample})
                # 获取输出并添加到结果
                batch_results.append(outputs[0])

            # 合并批次结果
            all_results.extend(batch_results)

        # 将所有结果堆叠为一个数组
        all_outputs = np.vstack([result for result in all_results])

        # 应用softmax获取概率
        probabilities = softmax(all_outputs, axis=1)

        # 计算推理时间
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒

        logger.info(f"推理完成，耗时: {inference_time:.2f} 毫秒")
        logger.info(f"预测结果形状: {probabilities.shape}")

        return probabilities, inference_time
    except Exception as e:
        logger.error(f"推理过程中发生错误: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="使用ONNX模型进行推理")
    parser.add_argument(
        "--model",
        type=str,
        default="./data/onnx_model/inception.onnx",
        help="ONNX模型路径",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test/validation_set.with_label.tfrecord-00000-of-00024.gz",
        help="测试数据路径",
    )

    args = parser.parse_args()

    # 检查文件路径是否存在
    if not os.path.exists(args.model):
        logger.error(f"模型文件不存在: {args.model}")
        return

    if not os.path.exists(args.test_data):
        logger.error(f"测试数据文件不存在: {args.test_data}")
        return

    try:
        # 加载ONNX模型
        session = load_onnx_model(args.model)

        # 加载测试数据
        images, labels = load_data_with_tfrecord(args.test_data)

        logger.info(f"对 {len(images)} 个样本进行推理")

        # 进行推理
        probabilities, inference_time = run_inference_onnx(session, images)

        # 输出推理结果统计
        logger.info(f"推理完成，总耗时: {inference_time:.2f} 毫秒")
        logger.info(f"平均每个样本耗时: {inference_time / len(images):.2f} 毫秒")

        # 计算预测类别
        predictions = np.argmax(probabilities, axis=1)

        # 计算准确率
        accuracy = np.mean(predictions == labels)
        logger.info(f"准确率: {accuracy:.4f}")

        # 计算混淆矩阵
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])

        # 生成分类报告
        report = classification_report(labels, predictions, digits=4)
        logger.info("\nONNX Classification Report:\n%s", report)

        # 绘制混淆矩阵并保存
        disp.plot()
        plt.title("ONNX Confusion Matrix")
        confusion_matrix_path = "data/onnx_confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300)
        logger.info(f"混淆矩阵已保存至: {confusion_matrix_path}")

    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
