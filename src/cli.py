import numpy as np

# project imports
from .config import SEED, DATA_PATH, set_global_seed, define_augmentation_params
from .data import load_npz_dataset, encode_labels, inspect_data, split_dataset
from .utils.cleaning import cleaned_dataset, info_Table
from .utils.viz import show_images
from .training import train_model_by_name, train_many



# ===================================================================================
#                                   Main
# ===================================================================================
def main():
	# fix randomness and hide warnings
	set_global_seed(SEED)
	
	# Load data
	images, labels = load_npz_dataset(DATA_PATH)
	binary_labels, _ = encode_labels(labels)

	# Data Inspection
	inspect_data(images, labels)
	show_images(images, binary_labels, 0, 60)

	# Data Cleaning and Inspection
	cleaned_images, unwanted_images, cleaned_indices, unwanted_indices = cleaned_dataset(
		images=images,
		color_distance_threshold=0.5,
		similarity_threshold=0.62,
	)
	
	# Create a boolean mask for the values to keep
	mask = np.ones(binary_labels.shape[0], dtype=bool)
	mask[cleaned_indices] = False
	cleaned_labels = binary_labels[mask]
	outlier_labels = binary_labels[cleaned_indices]

	print(" " * 25, end=" ")
	print("Unwanted Images ", end=' ' * 25)
	show_images(unwanted_images, outlier_labels, 0, 10)

	print(" " * 25, end=" ")
	print("Cleaned Images ", end=' ' * 25)
	show_images(cleaned_images, cleaned_labels, 0, 10)

	info_Table(cleaned_images, unwanted_images, cleaned_labels, outlier_labels)

	# Split dataset and Augmentation
	aug_params = define_augmentation_params()
	aug_train_set, validation_set, class_weights = split_dataset(cleaned_images, cleaned_labels, aug_params, SEED)

	# Train Model
	# A) Train one model
	history = train_model_by_name(
		"efficientnetb0",
		aug_train_set,
		validation_set,
		epochs=60,
		class_weight=class_weights,   # from split_dataset
	)

	# B) Train several models at once
	# results = train_many(
	#     ["resnet50", "inceptionv3", "mobilenetv2"],
	#     aug_train_set,
	#     validation_set,
	#     per_model_overrides={
	#         "resnet50": {"epochs": 50, "class_weight": class_weights, "hidden_units": 128},
	#         "inceptionv3": {"epochs": 60, "class_weight": class_weights},
	#         "mobilenetv2": {"epochs": 50, "class_weight": class_weights},
	#     },
	#     verbose=1,
	#     plot_each=False,
	# )



if __name__ == "__main__":
	raise SystemExit(main())
