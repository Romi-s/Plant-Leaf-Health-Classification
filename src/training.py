"""
A small, extensible training API for all models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Tuple
import inspect

# project imports
from .config import MODELS
from .models.resnet50 import build_resnet50_model, get_resnet50_callbacks
from .models.efficientnetb0 import build_efficientnetb0_model, get_efficientnetb0_callbacks
from .models.inceptionv3 import build_inceptionv3_model, get_inceptionv3_callbacks
from .models.cnn import build_cnn_model, get_cnn_callbacks
from .models.mobilenetv2 import build_mobilenetv2_model, get_mobilenetv2_callbacks


# --------------------------- Model registry definitions ---------------------------

@dataclass
class ModelSpec:
    builder: Callable[..., Any]                      # returns a compiled tf.keras.Model
    callbacks: Optional[Callable[[], list]] = None   # factory returning a list of callbacks


_REGISTRY: Dict[str, ModelSpec] = {}
_DEFAULTS: Dict[str, Dict[str, Any]] = {}  # e.g. {"cnn": {"epochs": 50, ...}}


def register_model(name: str, builder: Callable[..., Any], callbacks: Optional[Callable[[], list]] = None) -> None:
    """Register a model by name with its builder and optional callbacks factory."""
    key = name.lower()
    _REGISTRY[key] = ModelSpec(builder=builder, callbacks=callbacks)
    _DEFAULTS.setdefault(key, {})


def set_defaults(name: str, **defaults: Any) -> None:
    """Set per-model defaults such as epochs and builder kwargs (kept for completeness)."""
    key = name.lower()
    _DEFAULTS.setdefault(key, {})
    _DEFAULTS[key].update(defaults)


def get_defaults(name: str) -> Dict[str, Any]:
    return _DEFAULTS.get(name.lower(), {})


# --- Project registration ------------------------------------------------------
register_model("resnet50", build_resnet50_model, get_resnet50_callbacks)
register_model("efficientnetb0", build_efficientnetb0_model, get_efficientnetb0_callbacks)
register_model("inceptionv3", build_inceptionv3_model, get_inceptionv3_callbacks)
register_model("cnn", build_cnn_model, get_cnn_callbacks)
register_model("mobilenetv2", build_mobilenetv2_model, get_mobilenetv2_callbacks)

# Load per-model defaults from config (single source of truth)
for _name, _cfg in MODELS.items():
    set_defaults(_name, epochs=_cfg.training.epochs, **(_cfg.training.builder_kwargs or {}))


# --- Internal helpers -------------------------------------------------------------------
def _build_model_by_name(name: str, *, fit_context: Optional[Dict[str, Any]] = None) -> Tuple[Any, list]:
    """
    Create a compiled model and its callbacks by name.
    We pass `fit_context` only if the builder actually accepts it (checked via signature).
    """
    key = name.lower()
    spec = _REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"Unknown model '{name}'. Registered: {list(_REGISTRY.keys())}")

    builder = spec.builder
    callbacks_factory = spec.callbacks

    # Safer than catching TypeError: check if the builder has a 'fit_context' param.
    builder_params = inspect.signature(builder).parameters
    if "fit_context" in builder_params:
        model = builder(fit_context=fit_context)
    else:
        model = builder()

    callbacks = callbacks_factory() if callbacks_factory else []
    return model, callbacks


# --- Public training function -----------------------------------------------------------
def train_model_by_name(
    name: str,
    train_data: Any,
    val_data: Any,
    *,
    fit_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """
    Single API used by CLI and notebooks.
    Builds the model from the registry and fits it with the provided datasets.
    All training/tuning knobs (epochs, class weight policy, etc.) are sourced from config.MODELS.
    """
    # Build model + callbacks (tuning happens inside the CNN builder if enabled)
    model, callbacks = _build_model_by_name(name, fit_context=fit_context)

    # Resolve epochs & class-weight policy from config; fall back to _DEFAULTS if needed
    key = name.lower()
    cfg = MODELS.get(key)
    if cfg is not None:
        epochs = cfg.training.epochs
        # Optional flag in TrainingConfig; default to True if absent
        use_class_weight = getattr(cfg.training, "use_class_weight", True)
    else:
        defaults = get_defaults(key)
        epochs = defaults.get("epochs", 50)
        use_class_weight = True

    # Class weights can be passed via fit_context (computed in CLI/data layer)
    class_weight = fit_context.get("class_weight") if (use_class_weight and fit_context) else None

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    return model, history
