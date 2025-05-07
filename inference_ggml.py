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
    # Define TFRecord parsing rules
    description = {
        "image/encoded": "byte",
        "label": "int",
    }

    # Load data using TFRecordDataset
    dataset = TFRecordDataset(
        data_path,
        compression_type="gzip",
        index_path=None,
        description=description,
    )

    images = []
    labels = []

    for record in dataset:
        # Decode image
        image = np.frombuffer(record["image/encoded"], dtype=np.uint8).reshape(
            INPUT_SHAPE
        )
        # Convert dimension order: (height, width, channels) -> (channels, height, width)
        image = np.transpose(image, (2, 0, 1))
        # Normalize
        image = (image - 128.0) / 128.0
        # Get label
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
    Load GGML model

    Args:
        model_path: GGUF model file path

    Returns:
        Loaded model
    """
    logger.info(f"Loading model: {model_path}")
    try:
        model = InceptionV3GGML(model_path)
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def run_inference(model, image_array):
    """
    Run inference using GGML model

    Args:
        model: Loaded InceptionV3GGML model
        image_array: Preprocessed image data

    Returns:
        Inference results
    """
    logger.info(f"Starting inference on {image_array.shape[0]} samples")

    # Run inference
    start_time = datetime.now()
    try:
        outputs = model(image_array)
        outputs = softmax(outputs, axis=1)
        end_time = datetime.now()

        # Calculate inference time
        inference_time = (
            end_time - start_time
        ).total_seconds() * 1000  # Convert to milliseconds

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

    # Check if file paths exist
    if not os.path.exists(args.model):
        logger.error(f"Model file does not exist: {args.model}")
        return

    if not os.path.exists(args.test_data):
        logger.error(f"Test data file does not exist: {args.test_data}")
        return

    try:
        # Load model
        model = load_model(args.model)

        # Load test data
        images, labels = load_data_with_tfrecord(args.test_data)

        # Limit sample count
        if args.num_samples > 0 and args.num_samples < len(images):
            images = images[: args.num_samples]
            labels = labels[: args.num_samples]

        logger.info(f"Running inference on {len(images)} samples")

        # Run inference
        outputs, inference_time = run_inference(model, images)

        # Output inference results
        logger.info(
            f"Inference completed, total time: {inference_time:.2f} milliseconds"
        )
        logger.info(
            f"Average time per sample: {inference_time / len(images):.2f} milliseconds"
        )
        logger.info(
            f"Prediction output shape: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}"
        )

        # Calculate confusion matrix and classification report
        predictions = np.argmax(outputs, axis=1)
        m = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=[0, 1, 2])
        report = classification_report(labels, predictions, digits=4)
        logger.info("\nGGML Classification Report:\n%s", report)
        disp.plot()
        plt.title("GGML Confusion Matrix")
        plt.savefig("./data/ggml_confusion_matrix.png", dpi=300)

    except Exception as e:
        logger.error(f"Error occurred during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
