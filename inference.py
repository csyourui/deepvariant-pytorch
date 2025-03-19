import argparse
import logging

import matplotlib.pyplot as plt
import tensorflow as tf
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from tensorflow_model.keras_modeling import inceptionv3 as tf_inception_v3

INPUT_SHAPE = (100, 221, 7)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(args, mode="tf"):
    def preprocess_images(images):
        images = tf.cast(images, dtype=tf.float32)
        images = tf.subtract(images, 128.0)
        images = tf.math.divide(images, 128.0)
        return images

    ds = tf.data.TFRecordDataset(
        args.test_data,
        buffer_size=1024 * 1024 * 1024,
        compression_type="GZIP",
    )
    PROTO_FEATURES = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string),
        "label": tf.io.FixedLenFeature((1), tf.int64),
    }

    images = []
    labels = []

    for record in ds:
        result = tf.io.parse_single_example(serialized=record, features=PROTO_FEATURES)

        image = tf.io.decode_raw(result["image/encoded"], tf.uint8)
        image = tf.reshape(image, INPUT_SHAPE)
        image = preprocess_images(image)

        label = result["label"]

        if mode == "pt":
            image = image.numpy()
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            label = label.numpy()
            label = torch.from_numpy(label)
        images.append(image)
        labels.append(label)
        if args.count > 0 and len(images) >= args.count:
            break

    if mode == "pt":
        return torch.stack(images), torch.stack(labels)

    return tf.stack(images), tf.stack(labels)


def run_tf_model(args):
    tf.get_logger().setLevel("ERROR")

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        logger.info("Loading TensorFlow model...")
        model = tf_inception_v3(args.tf_weights)
        logger.info("Loading validation data...")
        images, labels = load_data(args, "tf")
        logger.info(f"Loaded {len(images)} samples")
        logger.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

        logger.info("Running predictions...")
        predictions = model.predict(images, batch_size=256)
        predicted_classes = tf.cast(tf.argmax(predictions, axis=1), tf.int64)

        m = confusion_matrix(labels, predicted_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=[0, 1, 2])
        report = classification_report(labels, predicted_classes, digits=4)
        logger.info("\nTensorFlow Classification Report:\n%s", report)
        disp.plot()
        plt.title("TensorFlow Confusion Matrix")
        plt.savefig("./data/tf_confusion_matrix.png", dpi=300)


def run_pt_model(args):
    logger.info("Loading PyTorch model...")
    pt_model = torch.load(args.pt_weights, weights_only=False).to("mps")
    pt_model.eval()
    logger.info("Loading validation data...")
    images, labels = load_data(args, "pt")
    logger.info(f"Loaded {len(images)} samples")
    logger.info(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    logger.info("Running predictions...")
    batchsize = 256
    predictions = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batchsize)):
            batch_images = images[i : i + batchsize].to("mps")
            outputs = pt_model(batch_images)
            outputs = softmax(outputs)
            prediction = torch.argmax(outputs, dim=1)
            predictions.extend(prediction.cpu().numpy())

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
    run_tf_model(args)
