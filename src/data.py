import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

from .config import BATCH_SIZE

def load_npz_dataset(path: str):
    dataset = np.load(path, allow_pickle=True)

    keys = list(dataset.keys())
    print('keys in our dataset are: ', keys)

    images = dataset["data"]
    labels = dataset["labels"]

    unique_classes, counts = np.unique(labels,return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} data points")

    return images, labels

def inspect_data(images, labels):
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

def encode_labels(labels):
    le = LabelEncoder()
    # Fit and transform the string labels to integer labels
    integer_labels = le.fit_transform(labels)
    # Convert integers to binary labels (0 or 1)
    binary_labels = (integer_labels == 1).astype(int)
    return binary_labels, le

def split_dataset(
    cleaned_images,
    cleaned_labels,
    num_classes: int = 2,
    aug_params: dict = None,
    seed: int = 42
):
    category_labels = to_categorical(cleaned_labels, num_classes)
    
    # Calculate class weights
    class_frequencies = np.sum(np.array(category_labels), axis=0)
    total_samples = np.sum(class_frequencies)
    class_weights = {i: total_samples / (len(class_frequencies) * freq) for i, freq in enumerate(class_frequencies)}

    if aug_params is None:
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
    data_datagen = ImageDataGenerator(**aug_params)

    aug_train_set = data_datagen.flow(
        cleaned_images,
        category_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        subset="training"
    )
    validation_set = data_datagen.flow(
        cleaned_images,
        category_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        subset="validation"
    )
    
    return aug_train_set, validation_set, class_weights
