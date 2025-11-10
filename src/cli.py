import argparse
from typing import List, Optional
import numpy as np

# project imports
from .config import SEED, DATA_PATH, set_global_seed, define_augmentation_params, MODELS
from .data import load_npz_dataset, encode_labels, inspect_data, split_dataset
from .utils.cleaning import cleaned_dataset, info_Table
from .utils.viz import show_images, plot_history
from .utils.metric import evaluate_model
from .training import train_model_by_name


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

    # Optional visualizations/evaluation
    ap.add_argument(
        "--plot-history",
        action="store_true",
        help="Plot training history curves if available.",
    )
    ap.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model(s) on validation set after training.",
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

    # Data inspection / visualization
    try:
        inspect_data(images, labels)
        show_images(images, binary_labels, 0, 60)
    except Exception:
        pass

    # Data cleaning
    cleaned_images, unwanted_images, cleaned_indices, unwanted_indices = cleaned_dataset(
        images=images,
        color_distance_threshold=0.5,
        similarity_threshold=0.62,
    )

    # Build label arrays using the correct indices
    cleaned_labels = binary_labels[np.array(cleaned_indices, dtype=int)]
    outlier_labels = binary_labels[np.array(unwanted_indices, dtype=int)]

    # Visualize results with correctly paired labels/indices (optional)
    try:
        print(" " * 25, end=" ")
        print("Unwanted Images ", end=" " * 25)
        show_images(unwanted_images, outlier_labels, 0, 10)

        print(" " * 25, end=" ")
        print("Cleaned Images ", end=" " * 25)
        show_images(cleaned_images, cleaned_labels, 0, 10)
    except Exception:
        pass

    # info_Table expects indices, not labels
    info_Table(cleaned_images, unwanted_images, np.array(cleaned_indices), np.array(unwanted_indices))

    # Split dataset and set up augmentation
    aug_params = define_augmentation_params()
    aug_train_set, validation_set, class_weights, category_labels = split_dataset(
        cleaned_images, cleaned_labels, aug_params, SEED
    )

    # Fit context shared with all builders (CNN uses it for tuning)
    fit_context = {
        "train_gen": aug_train_set,
        "val_gen": validation_set,
        "class_weight": class_weights,
    }

    # helper to train/eval/plot for one model
    def run_one(name: str):
        name = name.lower()
        print(f"\n>>> Training: {name}")
        model, history = train_model_by_name(
            name,
            aug_train_set,
            validation_set,
            fit_context=fit_context,
        )
        if args.plot_history:
            plot_history(history, title=f"{name} — History")
        if args.eval:
            evaluate_model(model, validation_set, category_labels)

    # --- Training modes ---------------------------------------------------------
    if args.mode == "single":
        run_one(args.name)

    elif args.mode == "subset":
        if not args.names:
            raise SystemExit("For --mode=subset you must provide --names (e.g., --names resnet50 inceptionv3 mobilenetv2)")
        for name in [n.lower() for n in args.names]:
            run_one(name)

    elif args.mode == "all":
        # Train every registered model from config
        for name in MODELS.keys():
            run_one(name)

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())