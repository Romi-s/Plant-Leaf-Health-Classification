import argparse
from typing import List, Optional
import numpy as np

# project imports
from .config import SEED, DATA_PATH, set_global_seed, define_augmentation_params
from .data import load_npz_dataset, encode_labels, inspect_data, split_dataset
from .utils.cleaning import cleaned_dataset, info_Table
from .utils.viz import show_images
from .training import train_model_by_name, train_many, MODEL_BUILDERS


def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser("Plant Classification — Training")

    ap.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "subset", "all"],
        help="Training mode: single | subset | all",
    )

    # For single-model runs
    ap.add_argument(
        "--name",
        type=str,
        default="resnet50",
        help="Model name for --mode=single (e.g., cnn, resnet50, efficientnetb0, inceptionv3, mobilenetv2)",
    )

    # For subset runs (multiple models)
    ap.add_argument(
        "--names",
        nargs="+",
        help="Model names for --mode=subset (space-separated). Example: --names resnet50 inceptionv3 mobilenetv2",
    )

    # Optional overrides
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for single run (keeps per-model defaults when omitted).",
    )
    return ap.parse_args(argv)


# ===================================================================================
#                                   Main
# ===================================================================================
def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # Fix randomness and hide warnings
    set_global_seed(SEED)

    # Load data
    images, labels = load_npz_dataset(DATA_PATH)
    binary_labels, _ = encode_labels(labels)

    # Data inspection
    inspect_data(images, labels)
    show_images(images, binary_labels, 0, 60)

    # Data cleaning
    cleaned_images, unwanted_images, cleaned_indices, unwanted_indices = cleaned_dataset(
        images=images,
        color_distance_threshold=0.5,
        similarity_threshold=0.62,
    )

    # Build label arrays using the correct indices
    cleaned_labels = binary_labels[np.array(cleaned_indices, dtype=int)]
    outlier_labels = binary_labels[np.array(unwanted_indices, dtype=int)]

    # Visualize results with correctly paired labels/indices
    print(" " * 25, end=" ")
    print("Unwanted Images ", end=" " * 25)
    show_images(unwanted_images, outlier_labels, 0, 10)

    print(" " * 25, end=" ")
    print("Cleaned Images ", end=" " * 25)
    show_images(cleaned_images, cleaned_labels, 0, 10)

    # info_Table expects indices, not labels
    info_Table(cleaned_images, unwanted_images, np.array(cleaned_indices), np.array(unwanted_indices))

    # Split dataset and set up augmentation
    aug_params = define_augmentation_params()
    aug_train_set, validation_set, class_weights = split_dataset(cleaned_images, cleaned_labels, aug_params, SEED)

    # --- Training modes ---------------------------------------------------------
    if args.mode == "single":
        history = train_model_by_name(
            args.name,
            aug_train_set,
            validation_set,
            epochs=args.epochs,         
            class_weight=class_weights,
        )

    elif args.mode == "subset":
        if not args.names:
            raise SystemExit("For --mode=subset you must provide --names (e.g., --names resnet50 inceptionv3 mobilenetv2)")
        results = train_many(
            [n.lower() for n in args.names],
            aug_train_set,
            validation_set,
            per_model_overrides={
                "resnet50": {"epochs": 50, "class_weight": class_weights, "hidden_units": 128},
                "inceptionv3": {"epochs": 60, "class_weight": class_weights},
                "mobilenetv2": {"epochs": 50, "class_weight": class_weights},
            },
            verbose=1,
            plot_each=False,
        )

    elif args.mode == "all":
        # Train every registered model
        all_names: List[str] = list(MODEL_BUILDERS.keys())
        results = train_many(
            all_names,
            aug_train_set,
            validation_set,
            per_model_overrides={
                "resnet50": {"epochs": 50, "class_weight": class_weights, "hidden_units": 128},
                "inceptionv3": {"epochs": 60, "class_weight": class_weights},
                "mobilenetv2": {"epochs": 50, "class_weight": class_weights},
                # Add overrides for other models as needed
            },
            verbose=1,
            plot_each=False,
        )

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
