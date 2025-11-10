"""Plant Leaf Health Classification – public API shortcuts."""
from .training import train_model_by_name, register_model, set_defaults

__all__ = ["train_model_by_name", "register_model", "set_defaults"]
__version__ = "0.2.0"