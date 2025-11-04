# src/models/cnn.py

from typing import Tuple

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

from src.config import SEED, IMG_SIZE, N_CHANNELS


def build_cnn(
    input_shape: Tuple[int, int, int] | None = None,
    n_classes: int = 2,
) -> tfk.Model:
    """
    Simple fixed CNN (no hyperparameter tuning).
    """
    if input_shape is None:
        input_shape = (*IMG_SIZE, N_CHANNELS)

    model = tfk.models.Sequential(
        [
            tfkl.Conv2D(
                64,
                (3, 3),
                activation="elu",
                input_shape=input_shape,
            ),
            tfkl.BatchNormalization(renorm=True),
            tfkl.MaxPooling2D(2, 2),

            # You can add more Conv/Pooling blocks here if you want
            # e.g.
            # tfkl.Conv2D(128, (3, 3), activation="elu"),
            # tfkl.BatchNormalization(renorm=True),
            # tfkl.MaxPooling2D(2, 2),

            tfkl.Flatten(),
            tfkl.Dense(256, activation="elu"),
            tfkl.Dense(n_classes, activation="softmax"),
        ]
    )
    return model


def build_cnn_tuned(hp) -> tfk.Model:
    """
    CNN model for Keras Tuner.

    `hp` is a keras_tuner.HyperParameters object.
    """
    input_shape = (*IMG_SIZE, N_CHANNELS)
    n_classes = 2  # adjust if you have more classes

    model = tfk.models.Sequential(
        [
            # ---- Conv Block 1 ----
            tfkl.Conv2D(
                filters=hp.Int("conv1_filters", min_value=32, max_value=128, step=16),
                kernel_size=hp.Choice("conv1_kernel", values=[3, 5]),
                activation="elu",
                input_shape=input_shape,
            ),
            tfkl.BatchNormalization(renorm=True),
            tfkl.MaxPooling2D(2, 2),

            # ---- Conv Block 2 ----
            tfkl.Conv2D(
                filters=hp.Int("conv2_filters", min_value=64, max_value=128, step=32),
                kernel_size=hp.Choice("conv2_kernel", values=[3, 5]),
                activation="elu",
            ),
            tfkl.Conv2D(
                filters=hp.Int("conv3_filters", min_value=64, max_value=256, step=32),
                kernel_size=hp.Choice("conv3_kernel", values=[3, 5]),
                activation="elu",
            ),
            tfkl.BatchNormalization(renorm=True),
            tfkl.MaxPooling2D(2, 2),

            # ---- Conv Block 3 ----
            tfkl.Conv2D(
                filters=hp.Int("conv4_filters", min_value=64, max_value=512, step=32),
                kernel_size=hp.Choice("conv4_kernel", values=[3, 5]),
                activation="elu",
            ),
            tfkl.Conv2D(
                filters=hp.Int("conv5_filters", min_value=128, max_value=1024, step=64),
                kernel_size=hp.Choice("conv5_kernel", values=[3, 5]),
                activation="elu",
            ),
            tfkl.BatchNormalization(renorm=True),

            tfkl.Conv2D(
                filters=hp.Int("conv6_filters", min_value=64, max_value=512, step=64),
                kernel_size=hp.Choice("conv6_kernel", values=[3, 5]),
                activation="elu",
            ),
            tfkl.Conv2D(
                filters=hp.Int("conv7_filters", min_value=32, max_value=256, step=16),
                kernel_size=hp.Choice("conv7_kernel", values=[3, 5]),
                activation="elu",
            ),

            tfkl.GlobalMaxPooling2D(),

            # GlobalMaxPooling already produces a 1D vector, Flatten is redundant,
            # but we'll keep it if you want to stay closer to the original:
            # tfkl.Flatten(),

            # ---- Dense / Dropout block ----
            tfkl.Dropout(
                rate=hp.Float(
                    "dropout1_rate",
                    min_value=0.05,
                    max_value=0.2,
                    step=0.05,
                ),
                seed=SEED,
            ),
            tfkl.Dense(
                units=hp.Int(
                    "dense1_units", min_value=64, max_value=512, step=32
                ),
                activation=None,
            ),

            tfkl.Dropout(
                rate=hp.Float(
                    "dropout2_rate",
                    min_value=0.4,
                    max_value=0.6,
                    step=0.1,
                ),
                seed=SEED,
            ),
            tfkl.Dense(
                units=hp.Int(
                    "dense2_units", min_value=32, max_value=256, step=16
                ),
                activation=None,
            ),

            # Dense + LeakyReLU must be split into two layers:
            tfkl.Dense(
                units=hp.Int(
                    "dense3_units", min_value=32, max_value=128, step=8
                ),
                activation=None,
            ),
            tfkl.LeakyReLU(
                alpha=hp.Float(
                    "leaky_relu_alpha",
                    min_value=0.01,
                    max_value=0.3,
                    step=0.01,
                )
            ),

            # Output layer
            tfkl.Dense(n_classes, activation="softmax"),
        ]
    )

    opt = tfk.optimizers.Adam(
        learning_rate=hp.Choice(
            "learning_rate",
            values=[1e-3, 5e-4, 1e-4],
        )
    )

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model