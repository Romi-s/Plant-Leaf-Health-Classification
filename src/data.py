import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_npz_dataset(path: str):
    dataset = np.load(path, allow_pickle=True)

    keys = list(dataset.keys())
    print('keys in our dataset are: ', keys)

    images = dataset["images"]
    labels = dataset["labels"]

    unique_classes, counts = np.unique(labels,return_counts=True)
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} data points")

    return images, labels

def encode_labels(labels):
    le = LabelEncoder()
    # Fit and transform the string labels to integer labels
    integer_labels = le.fit_transform(labels)
    # Convert integers to binary labels (0 or 1)
    binary_labels = (integer_labels == 1).astype(int)
    return binary_labels, le

def split_dataset(X, y, test_size=0.2, val_size=0.2, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    rel_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=random_state
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
