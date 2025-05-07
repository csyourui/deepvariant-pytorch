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
    # Define the parsing rules for TFRecord
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
        # Decode the image
        image = np.frombuffer(record["image/encoded"], dtype=np.uint8).reshape(
            INPUT_SHAPE
        )
        # Normalize
        image = (image - 128.0) / 128.0
        # Get the label
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
    Load ONNX model

    Args:
        model_path: Path to the ONNX model file

    Returns:
        Loaded ONNX model session
    """
    logger.info(f"Loading ONNX model: {model_path}")
    try:
        # Set ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Create inference session
        session = ort.InferenceSession(model_path, sess_options)
        logger.info("ONNX model loaded successfully")

        # Output model input and output information
        input_details = session.get_inputs()
        output_details = session.get_outputs()
        logger.info(f"Model inputs: {[input.name for input in input_details]}")
        logger.info(f"Model input shapes: {[input.shape for input in input_details]}")
        logger.info(f"Model outputs: {[output.name for output in output_details]}")

        return session
    except Exception as e:
        logger.error(f"Failed to load ONNX model: {e}")
        raise


def prepare_batch_input(images, batch_size=256):
    """
    Prepare batch input data, adjusting to the format expected by the ONNX model

    Args:
        images: Array of image data
        batch_size: Batch size

    Returns:
        List of batches, each with shape [batch_size, height, width, channels]
    """
    num_samples = len(images)
    batches = []

    for i in range(0, num_samples, batch_size):
        end = min(i + batch_size, num_samples)
        batch_images = images[i:end]
        batches.append(batch_images)

    return batches


def run_inference_onnx(session, image_array):
    """
    Run inference using ONNX model

    Args:
        session: ONNX Runtime session
        image_array: Preprocessed image data (N, H, W, C)

    Returns:
        Inference results and inference time
    """
    logger.info(f"Starting inference for {len(image_array)} samples")

    # Get model input name
    input_name = session.get_inputs()[0].name

    # Record start time
    start_time = time.time()

    try:
        # Prepare batch data
        batches = prepare_batch_input(image_array)
        all_results = []

        for batch in batches:
            # Run inference on the entire batch
            outputs = session.run(None, {input_name: batch})
            # Add batch results to total results
            all_results.append(outputs[0])

        # Merge all results
        all_outputs = np.vstack(all_results)

        # Apply softmax to get probabilities
        probabilities = softmax(all_outputs, axis=1)

        # Calculate inference time
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

        logger.info(f"Inference completed, time taken: {inference_time:.2f} ms")
        logger.info(f"Prediction results shape: {probabilities.shape}")

        return probabilities, inference_time
    except Exception as e:
        logger.error(f"Error occurred during inference: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Use ONNX model for inference")
    parser.add_argument(
        "--model",
        type=str,
        default="./data/onnx_model/inception.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data/test/validation_set.with_label.tfrecord-00000-of-00024.gz",
        help="Path to test data",
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
        # Load ONNX model
        session = load_onnx_model(args.model)

        # Load test data
        images, labels = load_data_with_tfrecord(args.test_data)

        logger.info(f"Running inference on {len(images)} samples")

        # Perform inference
        probabilities, inference_time = run_inference_onnx(session, images)

        # Output inference result statistics
        logger.info(f"Inference completed, total time: {inference_time:.2f} ms")
        logger.info(f"Average time per sample: {inference_time / len(images):.2f} ms")

        # Calculate predicted classes
        predictions = np.argmax(probabilities, axis=1)

        # Calculate accuracy
        accuracy = np.mean(predictions == labels)
        logger.info(f"Accuracy: {accuracy:.4f}")

        # Calculate confusion matrix
        cm = confusion_matrix(labels, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1, 2])

        # Generate classification report
        report = classification_report(labels, predictions, digits=4)
        logger.info("\nONNX Classification Report:\n%s", report)

        # Plot and save confusion matrix
        disp.plot()
        plt.title("ONNX Confusion Matrix")
        confusion_matrix_path = "data/onnx_confusion_matrix.png"
        plt.savefig(confusion_matrix_path, dpi=300)
        logger.info(f"Confusion matrix saved to: {confusion_matrix_path}")

    except Exception as e:
        logger.error(f"Error occurred during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
