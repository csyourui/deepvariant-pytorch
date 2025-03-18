# modify from https://github.com/google/deepvariant/blob/r1.8/deepvariant/keras_modeling.py

from typing import Type

import tensorflow as tf

NUM_CLASSES = 3
DEFAULT_WEIGHT_DECAY = 0.00004
DEFAULT_BACKBONE_DROPOUT_RATE = 0.2
INPUT_SHAPE = (100, 221, 7)


def inceptionv3(weights):
    tf.get_logger().setLevel("ERROR")

    def build_classification_head(inputs: tf.Tensor, l2: float = 0.0) -> tf.Tensor:
        l2_regularizer = tf.keras.regularizers.L2(l2) if l2 else None
        head = tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax",
            dtype=tf.float32,
            name="classification",
            kernel_regularizer=l2_regularizer,
        )
        return head(inputs)

    def add_l2_regularizers(
        model: tf.keras.Model,
        layer_class: Type[tf.keras.layers.Layer],
        l2: float = DEFAULT_WEIGHT_DECAY,
    ) -> tf.keras.Model:
        if not l2:
            return model
        num_regularizers_added = 0

        def add_l2_regularization(layer):
            def _add_l2():
                l2_reg = tf.keras.regularizers.l2(l2=l2)
                return l2_reg(layer.kernel)

            return _add_l2

        for layer in model.layers:
            if isinstance(layer, layer_class):
                model.add_loss(add_l2_regularization(layer))
                num_regularizers_added += 1

        return model

    backbone = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_shape=INPUT_SHAPE,
        classes=NUM_CLASSES,
        pooling="avg",
    )
    weight_decay = DEFAULT_WEIGHT_DECAY
    backbone_dropout_rate = DEFAULT_BACKBONE_DROPOUT_RATE
    hid = tf.keras.layers.Dropout(backbone_dropout_rate)(backbone.output)
    outputs = []
    outputs.append(build_classification_head(hid, l2=weight_decay))
    model = tf.keras.Model(inputs=backbone.input, outputs=outputs, name="inceptionv3")
    model = add_l2_regularizers(model, tf.keras.layers.Conv2D, l2=weight_decay)
    model.load_weights(weights)
    # model.summary()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    args = parser.parse_args()
    model = inceptionv3(args.weights)
    model.summary()
