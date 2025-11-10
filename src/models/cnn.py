from __future__ import annotations
from typing import Dict, Optional, Tuple
from tensorflow import keras as tfk
from keras import layers as tfkl
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from src.config import IMAGE_SHAPE, SEED, get_tuner_config, get_training_config
from src.utils.tuner import run_cnn_tuner

IMG_SIZE = (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
N_CHANNELS = IMAGE_SHAPE[2]

def _build_cnn_plain(
    *,
    input_shape: Optional[Tuple[int, int, int]] = None,
    n_classes: int = 2,
    conv1_filters: int = 64,
    conv2_filters: int = 128,
    conv3_filters: int = 256,
    kernel_size: int = 3,
    dense_units: int = 256,
    dropout_rate: float = 0.3,
) -> tfk.Model:
    if input_shape is None:
        input_shape = (*IMG_SIZE, N_CHANNELS)

    inputs = tfk.Input(shape=input_shape)

    # Block 1
    x = tfkl.Conv2D(conv1_filters, kernel_size, padding="same", activation="elu")(inputs)
    x = tfkl.BatchNormalization(renorm=True)(x)
    x = tfkl.MaxPooling2D(2, 2)(x)

    # Block 2
    x = tfkl.Conv2D(conv2_filters, kernel_size, padding="same", activation="elu")(x)
    x = tfkl.BatchNormalization(renorm=True)(x)
    x = tfkl.MaxPooling2D(2, 2)(x)

    # Block 3
    x = tfkl.Conv2D(conv3_filters, kernel_size, padding="same", activation="elu")(x)
    x = tfkl.BatchNormalization(renorm=True)(x)
    x = tfkl.MaxPooling2D(2, 2)(x)

    x = tfkl.GlobalAveragePooling2D()(x)
    x = tfkl.Dropout(dropout_rate)(x)
    x = tfkl.Dense(dense_units, activation="elu")(x)
    outputs = tfkl.Dense(n_classes, activation="softmax")(x)

    model = tfk.Model(inputs, outputs, name="cnn_plain")

    opt = tfk.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_tuned(
    hp,  # keras_tuner.HyperParameters
    input_shape: Optional[Tuple[int, int, int]] = None,
    n_classes: int = 2,
) -> tfk.Model:
    if input_shape is None:
        input_shape = (*IMG_SIZE, N_CHANNELS)

    conv1_filters = hp.Int("conv1_filters", min_value=32, max_value=128, step=16)
    conv2_filters = hp.Int("conv2_filters", min_value=64, max_value=256, step=32)
    conv3_filters = hp.Int("conv3_filters", min_value=64, max_value=384, step=32)
    kernel_size = hp.Choice("kernel_size", values=[3, 5])
    dense_units = hp.Int("dense_units", min_value=128, max_value=512, step=64)
    dropout_rate = hp.Choice("dropout_rate", values=[0.2, 0.3, 0.4, 0.5])
    lr = hp.Choice("lr", values=[1e-3, 5e-4, 1e-4])

    inputs = tfk.Input(shape=input_shape)

    # Block 1
    x = tfkl.Conv2D(conv1_filters, kernel_size, padding="same", activation="elu")(inputs)
    x = tfkl.BatchNormalization(renorm=True)(x)
    x = tfkl.MaxPooling2D(2, 2)(x)

    # Block 2
    x = tfkl.Conv2D(conv2_filters, kernel_size, padding="same", activation="elu")(x)
    x = tfkl.BatchNormalization(renorm=True)(x)
    x = tfkl.MaxPooling2D(2, 2)(x)

    # Block 3
    x = tfkl.Conv2D(conv3_filters, kernel_size, padding="same", activation="elu")(x)
    x = tfkl.BatchNormalization(renorm=True)(x)
    x = tfkl.MaxPooling2D(2, 2)(x)

    x = tfkl.GlobalAveragePooling2D()(x)
    x = tfkl.Dropout(dropout_rate)(x)
    x = tfkl.Dense(dense_units, activation="elu")(x)
    outputs = tfkl.Dense(n_classes, activation="softmax")(x)

    model = tfk.Model(inputs, outputs, name="cnn_tuned_candidate")

    opt = tfk.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_model(
    *,
    input_shape: Optional[Tuple[int, int, int]] = None,
    n_classes: int = 2,
    fit_context: Optional[Dict[str, object]] = None,
    **builder_overrides,
) -> tfk.Model:
    """
    Generic CNN builder that optionally tunes internally.

    Args:
        input_shape: (H, W, C). Defaults to config IMG_SIZE/N_CHANNELS.
        n_classes: number of classes.
        fit_context: optional dict with {'train_gen', 'val_gen', 'class_weight'}.
                     If provided AND tuning is enabled in config, tuner will run.
        **builder_overrides: forwards into the plain CNN builder (filters, units, etc.)
    """
    # Read config
    train_cfg = get_training_config("cnn")
    tuner_cfg = get_tuner_config("cnn")

    # If tuning is enabled and we have data, run tuner to get the best model.
    if tuner_cfg.enabled and fit_context is not None:
        train_gen = fit_context.get("train_gen")
        val_gen = fit_context.get("val_gen")

        if train_gen is None or val_gen is None:
            # No usable data in context; fall back to plain
            return _build_cnn_plain(
                input_shape=input_shape,
                n_classes=n_classes,
                **builder_overrides,
                **(train_cfg.builder_kwargs or {}),
            )

        best_model, _ = run_cnn_tuner(
            build_cnn_tuned,
            train_gen,
            val_gen,
            epochs=tuner_cfg.epochs_per_trial,
            max_trials=tuner_cfg.max_trials,
            project_name=tuner_cfg.project_name,
            directory=tuner_cfg.directory,
            objective=tuner_cfg.objective or "val_accuracy",
        )
        return best_model

    # Otherwise: plain CNN using merged overrides + config defaults
    return _build_cnn_plain(
        input_shape=input_shape,
        n_classes=n_classes,
        **(train_cfg.builder_kwargs or {}),
        **builder_overrides,
    )


def get_cnn_callbacks():
	lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.3, min_lr=0.000001)
	checkpoint = ModelCheckpoint('model_CNN.hdf5', save_best_only=True, monitor='val_loss', mode='min')
	return [lr_reduction, checkpoint]
