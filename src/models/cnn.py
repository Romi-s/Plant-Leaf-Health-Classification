from typing import Optional, Tuple
from tensorflow import keras as tfk
from keras import layers as tfkl
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from src.config import IMG_SIZE, N_CHANNELS, SEED


def build_cnn_tuned(
    hp,
    input_shape: Optional[Tuple[int, int, int]] = None,
    n_classes: int = 2,
) -> tfk.Model:
    """
    Build a CNN model compatible with Keras Tuner.

    Args:
        hp: keras_tuner.HyperParameters object.
        input_shape: shape of input images (H, W, C). Defaults to config IMG_SIZE/N_CHANNELS.
        n_classes: number of output classes. Defaults to 2.

    Returns:
        Compiled tf.keras.Model
    """
    if input_shape is None:
        input_shape = (*IMG_SIZE, N_CHANNELS)

    model = tfk.models.Sequential([
        # ---- Conv Block 1 ----
        tfkl.Conv2D(
            filters=hp.Int("conv1_filters", min_value=32, max_value=128, step=16),
            kernel_size=hp.Choice("conv1_kernel", values=[3, 5]),
            activation="elu",
            input_shape=input_shape,
        ),
        tfkl.BatchNormalization(),  # 'renorm' can be fragile across TF versions
        tfkl.MaxPooling2D(pool_size=2, strides=2),

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
        tfkl.BatchNormalization(),
        tfkl.MaxPooling2D(pool_size=2, strides=2),

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
        tfkl.BatchNormalization(),

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

        # ---- Dense / Dropout block ----
        tfkl.Dropout(
            rate=hp.Float("dropout1_rate", min_value=0.05, max_value=0.2, step=0.05),
            seed=SEED,
        ),
        tfkl.Dense(
            units=hp.Int("dense1_units", min_value=64, max_value=512, step=32),
            activation=None,
        ),

        tfkl.Dropout(
            rate=hp.Float("dropout2_rate", min_value=0.4, max_value=0.6, step=0.1),
            seed=SEED,
        ),
        tfkl.Dense(
            units=hp.Int("dense2_units", min_value=32, max_value=256, step=16),
            activation=None,
        ),

        # Dense + LeakyReLU must be split into two layers
        tfkl.Dense(
            units=hp.Int("dense3_units", min_value=32, max_value=128, step=8),
            activation=None,
        ),
        tfkl.LeakyReLU(
            alpha=hp.Float("leaky_relu_alpha", min_value=0.01, max_value=0.3, step=0.01)
        ),

        # Output
        tfkl.Dense(n_classes, activation="softmax"),
    ])

    opt = tfk.optimizers.Adam(
        learning_rate=hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])
    )

    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_cnn_callbacks():
	lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.3, min_lr=0.000001)
	checkpoint = ModelCheckpoint('model_CNN.hdf5', save_best_only=True, monitor='val_loss', mode='min')
	return [lr_reduction, checkpoint]
