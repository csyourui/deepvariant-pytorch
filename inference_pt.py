import argparse
import logging

from tfrecord.torch.dataset import TFRecordDataset
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

INPUT_SHAPE = (100, 221, 7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data_with_tfrecord(args):
    # 定义 TFRecord 的解析规则
    description = {
        "image/encoded": "byte",
        "label": "int",
    }

    # 使用 TFRecordDataset 加载数据
    dataset = TFRecordDataset(
        args.test_data,
        compression_type="gzip",
        index_path=None, 
        description=description)

    images = []
    labels = []

    for record in dataset:
        # 解码图像
        image = torch.tensor(bytearray(record["image/encoded"]), dtype=torch.uint8)
        # 获取标签
        label = torch.tensor(record["label"], dtype=torch.int64)

        images.append(image)
        labels.append(label)

    # 将所有数据堆叠为 PyTorch 张量
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    # 将图像数据转换为 PyTorch 张量
    images = images.view(-1, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])
    # 将图像数据转换为 float32 类型
    images = images.to(torch.float32)
    # 图像数据归一化
    images = (images - 128.0) / 128.0  # 归一化
    # 将图像数据的通道维度调整到第二个维度，pytorch的通道维度在第二个维度
    images = images.permute(0, 3, 1, 2)  # 将通道维度调整到第二个维度

    return images, labels

def run_pt_model(args):
    logger.info("Loading PyTorch model...")
    pt_model = torch.load(args.pt_weights, weights_only=False)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    pt_model.to(device)
    pt_model.eval()
    logger.info("Loading validation data...")
    images, labels = load_data_with_tfrecord(args)
    logger.info(f"Loaded {len(images)} samples")
    logger.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    logger.info("Running predictions...")
    batchsize = 256
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batchsize)):
            batch_images = images[i : i + batchsize].to(device)
            outputs = pt_model(batch_images)
            softmax = torch.nn.Softmax(dim=1)
            outputs = softmax(outputs)
            prediction = torch.argmax(outputs, dim=1).to("cpu")
            predictions.extend(prediction.numpy())

    m = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=[0, 1, 2])
    report = classification_report(labels, predictions, digits=4)
    logger.info("\nPyTorch Classification Report:\n%s", report)
    disp.plot()
    plt.title("PyTorch Confusion Matrix")
    plt.savefig("./data/pt_confusion_matrix.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tf_weights",
        type=str,
        default="./data/tf_model/deepvariant.wgs.ckpt",
        help="TensorFlow model weights",
    )
    parser.add_argument(
        "--pt_weights",
        type=str,
        default="./data/pt_model/deepvariant.pt",
        help="PyTorch model weights",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test/validation_set.with_label.tfrecord-00000-of-00024.gz",
        help="Test data",
    )
    parser.add_argument(
        "--count", type=int, default=-1, help="Number of samples to load, -1 for all"
    )

    args = parser.parse_args()

    logger.info("Parameters:")
    logger.info("TensorFlow model weights: %s", args.tf_weights)
    logger.info("PyTorch model weights: %s", args.pt_weights)
    logger.info("Test data: %s", args.test_data)
    logger.info("Count: %d", args.count)
    run_pt_model(args)
