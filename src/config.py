"""
Fix randomness and hide warnings
Manage Augmentation Parameters (toggle or edit values)
Manage Training & Tuning Parameters (per-model)
"""

from __future__ import annotations

import logging
import os
import warnings
import random
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf

# ----------------------- Project-wide basics -----------------------

SEED: int = 42
BATCH_SIZE: int = 64
IMAGE_SHAPE = (96, 96, 3)          # (H, W, C)
IMG_SIZE = IMAGE_SHAPE[:2]         # (H, W)
N_CHANNELS = IMAGE_SHAPE[2]
DATA_PATH = "data/plant_dataset.npz"

# Toggle all image augmentations in one place
AUGMENTATION_ENABLED: bool = True


def set_global_seed(seed: int = SEED) -> None:
    # Python & NumPy
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # TF: determinism + low verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf.get_logger().setLevel(logging.ERROR)
    tf.autograph.set_verbosity(0)
    tf.random.set_seed(seed)

    # Optional (legacy) – safe to keep, but not required on TF2+
    try:
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    except Exception:
        pass

    # Matplotlib cache dir (avoid permission issues on some environments)
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "configs"))

    # Quiet common warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=Warning)


def define_augmentation_params() -> Dict[str, Any]:
    """
    Default augmentation parameters. If AUGMENTATION_ENABLED is False, returns an
    empty dict so callers can skip augmentation cleanly.
    """
    if not AUGMENTATION_ENABLED:
        return {}

    return {
        "rotation_range": 30,
        "zoom_range": [0.7, 1.3],
        "horizontal_flip": True,
        "vertical_flip": True,
        "brightness_range": [0.8, 1.2],
        "fill_mode": "reflect",
        "rescale": 1.0 / 255.0,
        "validation_split": 0.1,
        "preprocessing_function": None,  # override per model if you need preprocess_input
    }


# --------------------- Training & Tuning structures ---------------------

@dataclass
class TrainingConfig:
    epochs: int = 50
    # Whether trainer should pass class_weight to model.fit
    use_class_weight: bool = True
    # Anything you want to forward to the model builder (model-specific knobs)
    builder_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TunerConfig:
    enabled: bool = False           # if True, use Keras Tuner path for this model
    max_trials: int = 10
    epochs_per_trial: int = 30
    directory: str = "tuner_results"
    project_name: str = "hp_search"
    objective: Optional[str] = "val_accuracy"  # e.g., "val_accuracy" or "val_loss"


@dataclass
class ModelConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    tuner: TunerConfig = field(default_factory=TunerConfig)


# --------------------- Per-model defaults (edit here) ---------------------

MODELS: Dict[str, ModelConfig] = {
    "resnet50": ModelConfig(
        training=TrainingConfig(
            epochs=50,
            use_class_weight=True,
            builder_kwargs={
                "hidden_units": 128,
            },
        ),
        tuner=TunerConfig(enabled=False),
    ),
    "efficientnetb0": ModelConfig(
        training=TrainingConfig(
            epochs=60,
            use_class_weight=True,
            builder_kwargs={
                "hidden_units": 512,
                "hidden_units_1": 256,
            },
        ),
        tuner=TunerConfig(enabled=False),
    ),
    "inceptionv3": ModelConfig(
        training=TrainingConfig(
            epochs=60,
            use_class_weight=True,
            builder_kwargs={},
        ),
        tuner=TunerConfig(enabled=False),
    ),
    "mobilenetv2": ModelConfig(
        training=TrainingConfig(
            epochs=50,
            use_class_weight=True,
            builder_kwargs={},
        ),
        tuner=TunerConfig(enabled=False),
    ),

    # ---- CNN: tuner-ready ---------------------------------
    "cnn": ModelConfig(
        training=TrainingConfig(
            epochs=50,                # epochs for final training run after tuning
            use_class_weight=True,
            builder_kwargs={},        # hp comes from tuner if enabled
        ),
        tuner=TunerConfig(
            enabled=False,            # flip to True to use tuner by default
            max_trials=12,
            epochs_per_trial=25,
            directory="tuner_out",
            project_name="cnn_hp",
            objective="val_accuracy",
        ),
    ),
}


# --------------------------- Small helpers ---------------------------

def get_training_config(name: str) -> TrainingConfig:
    return MODELS[name.lower()].training

def get_tuner_config(name: str) -> TunerConfig:
    return MODELS[name.lower()].tuner
