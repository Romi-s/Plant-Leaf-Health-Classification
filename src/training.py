"""
A small, extensible training API for all models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Tuple

# project imports
from .utils.viz import plot_history
from .models.resnet50 import build_resnet50_model, get_resnet50_callbacks
from .models.efficientnetb0 import build_efficientnetb0_model, get_efficientnetb0_callbacks
from .models.inceptionv3 import build_inceptionv3_model, get_inceptionv3_callbacks
from .models.cnn import build_cnn_tuned, get_cnn_callbacks
from .models.mobilenetv2 import build_mobilenetv2_model, get_mobilenetv2_callbacks


# --- Protocols -----------------------------------------------------------------
class BuildFn(Protocol):
    def __call__(self, **kwargs: Any):
        """Return a **compiled** Keras model.
        The implementation may accept arbitrary kwargs (e.g., hidden_units, hp for Keras Tuner).
        """

class CallbacksFn(Protocol):
    def __call__(self) -> Optional[List[Any]]:
        """Return a list of Keras callbacks (or None)."""


# --- Registries ----------------------------------------------------------------
MODEL_BUILDERS: Dict[str, BuildFn] = {}
CALLBACK_FACTORIES: Dict[str, CallbacksFn] = {}


def register_model(name: str, build_fn: BuildFn, callbacks_fn: Optional[CallbacksFn] = None) -> None:
    """Register a model and its callbacks in the global registries."""
    key = name.lower()
    MODEL_BUILDERS[key] = build_fn
    if callbacks_fn is not None:
        CALLBACK_FACTORIES[key] = callbacks_fn


# --- Per‑model defaults ---------------------------------------------------------
@dataclass
class ModelDefaults:
    epochs: int = 50
    builder_kwargs: Dict[str, Any] = field(default_factory=dict)

MODEL_DEFAULTS: Dict[str, ModelDefaults] = {}


def set_defaults(name: str, *, epochs: Optional[int] = None, **builder_kwargs: Any) -> None:
    key = name.lower()
    cur = MODEL_DEFAULTS.get(key, ModelDefaults())
    if epochs is not None:
        cur.epochs = epochs
    if builder_kwargs:
        cur.builder_kwargs.update(builder_kwargs)
    MODEL_DEFAULTS[key] = cur


# --- Trainer -------------------------------------------------------------------
class Trainer:
    def __init__(self, *, model_name: str):
        key = model_name.lower()
        if key not in MODEL_BUILDERS:
            raise KeyError(f"Model '{model_name}' is not registered. Use register_model(name, build_fn, callbacks_fn).")
        self.model_name = key

    def _resolve_defaults(self, epochs: Optional[int], builder_kwargs: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        d = MODEL_DEFAULTS.get(self.model_name, ModelDefaults())
        resolved_epochs = epochs if epochs is not None else d.epochs
        merged_kwargs = {**d.builder_kwargs, **(builder_kwargs or {})}
        return resolved_epochs, merged_kwargs

    def _get_callbacks(self) -> Optional[List[Any]]:
        cb_fn = CALLBACK_FACTORIES.get(self.model_name)
        return cb_fn() if cb_fn else None

    def train(
        self,
        train_gen,
        val_gen,
        *,
        epochs: Optional[int] = None,
        verbose: int = 1,
        plot: bool = True,
        class_weight: Optional[dict] = None,
        **builder_kwargs: Any,
    ):
        """Build the model, attach callbacks, train, and (optionally) plot history.

        Example:
            Trainer(model_name="efficientnetb0").train(train_gen, val_gen, epochs=60,
                hidden_units=512, hidden_units_1=256)
        """
        epochs, builder_kwargs = self._resolve_defaults(epochs, builder_kwargs)
        model = MODEL_BUILDERS[self.model_name](**builder_kwargs)
        callbacks = self._get_callbacks()
        history = model.fit(
            train_gen,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            validation_data=val_gen,
            class_weight=class_weight,
        )
        if plot:
            plot_history(history)
        return history


# --- Project registration ------------------------------------------------------
register_model("resnet50", build_resnet50_model, get_resnet50_callbacks)
register_model("efficientnetb0", build_efficientnetb0_model, get_efficientnetb0_callbacks)
register_model("inceptionv3", build_inceptionv3_model, get_inceptionv3_callbacks)
register_model("cnn", build_cnn_tuned, get_cnn_callbacks)
register_model("mobilenetv2", build_mobilenetv2_model, get_mobilenetv2_callbacks)

# Sensible defaults (override at call time as needed)
set_defaults("resnet50", epochs=50, hidden_units=128)
set_defaults("efficientnetb0", epochs=60, hidden_units=512, hidden_units_1=256)
set_defaults("inceptionv3", epochs=60)
set_defaults("cnn", epochs=50)  # pass hp=... when calling Trainer.train
set_defaults("mobilenetv2", epochs=50)

# --- Convenience: train a single model by name --------------------------------

def train_model_by_name(
    name: str,
    train_gen,
    val_gen,
    *,
    epochs: Optional[int] = None,
    verbose: int = 1,
    plot: bool = True,
    class_weight: Optional[dict] = None,
    **builder_kwargs: Any,
):
    """One‑liner replacement for all your previous `train_*` functions.

    Example:
        history = train_model_by_name(
            "efficientnetb0", train_gen, val_gen, epochs=70, hidden_units=1024
        )
    """
    return Trainer(model_name=name).train(
        train_gen,
        val_gen,
        epochs=epochs,
        verbose=verbose,
        plot=plot,
        class_weight=class_weight,
        **builder_kwargs,
    )


# --- Bonus: train many models in a loop ---------------------------------------

def train_many(
    names: Iterable[str],
    train_gen,
    val_gen,
    *,
    per_model_overrides: Optional[Mapping[str, Dict[str, Any]]] = None,
    verbose: int = 1,
    plot_each: bool = False,
) -> Dict[str, Any]:
    """Train multiple registered models and return a dict name ➜ history.

    `per_model_overrides` allows per‑model kwargs/epochs, e.g.:
        per_model_overrides={
            "resnet50": {"epochs": 40, "hidden_units": 256},
            "cnn": {"epochs": 25, "hp": hp},
        }
    """
    results: Dict[str, Any] = {}
    for name in names:
        overrides = (per_model_overrides or {}).get(name.lower(), {}).copy()
        epochs = overrides.pop("epochs", None)
        class_weight = overrides.pop("class_weight", None)
        history = Trainer(model_name=name).train(
            train_gen,
            val_gen,
            epochs=epochs,
            verbose=verbose,
            plot=plot_each,
            class_weight=class_weight,
            **overrides,
        )
        results[name.lower()] = history
    return results
