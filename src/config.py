"""Fix randomness and hide warnings"""

import logging
import os, warnings
import random
import numpy as np
import tensorflow as tf

SEED = 42
BATCH_SIZE = 64
IMAGE_SHAPE = (96, 96, 3)
IMG_SIZE = (IMAGE_SHAPE[0], IMAGE_SHAPE[1])  # (96, 96)
N_CHANNELS = IMAGE_SHAPE[2]
DATA_PATH = "data/plant_dataset.npz"

def set_global_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
    random.seed(seed)
    np.random.seed(seed)

    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.random.set_seed(seed)
    tf.compat.v1.set_random_seed(seed)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)


def define_augmentation_params():
    """Define default augmentation parameters for ImageDataGenerator."""
    aug_params = {
        "rotation_range": 30,
        "zoom_range": [0.7, 1.3],
        "horizontal_flip": True,
        "vertical_flip": True,
        "brightness_range": [0.8, 1.2],
        "fill_mode": "reflect",
        "rescale": 1.0 / 255,
        "validation_split": 0.1,
        "preprocessing_function": None
    }
    return aug_params