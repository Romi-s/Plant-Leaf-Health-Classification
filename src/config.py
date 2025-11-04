"""Fix randomness and hide warnings"""

import logging
import os, warnings
import random
import numpy as np
import tensorflow as tf

SEED = 42
IMG_SIZE = (96, 96)
N_CHANNELS = 3

def set_global_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'
    random.seed(random.seed)
    np.random.seed(seed)

    tf.autograph.set_verbosity(0)
    tf.get_logger().setLevel(logging.ERROR)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.random.set_seed(SEED)
    tf.compat.v1.set_random_seed(SEED)

    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)