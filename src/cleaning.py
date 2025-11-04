from typing import List, Tuple, Sequence

import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76


def compute_histogram(image: np.ndarray) -> np.ndarray:
    """
    Compute a normalized 3D color histogram for an RGB image.
    Bins: 8x8x8 over [0, 256] per channel.
    """
    hist = cv2.calcHist(
        [image],
        channels=[0, 1, 2],
        mask=None,
        histSize=[8, 8, 8],
        ranges=[0, 256, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist)
    return hist


def is_histogram_different(
    image: np.ndarray,
    reference_histogram: np.ndarray,
    threshold: float = 0.8,
) -> bool:
    """
    Compare image histogram to a reference histogram.
    Returns True if the intersection is BELOW the threshold (i.e. 'different').

    threshold close to 1.0 → allow only very similar images.
    """
    image_histogram = compute_histogram(image)
    intersection = cv2.compareHist(
        reference_histogram, image_histogram, cv2.HISTCMP_INTERSECT
    )
    return intersection < threshold


def calculate_color_distance(
    image1: np.ndarray, image2: np.ndarray
) -> np.ndarray:
    """
    Calculate the per-pixel color distance between two RGB images using the CIE76 metric.
    Both images are expected as HxWx3, uint8 or float in [0, 1].
    """
    # Ensure float in [0, 1] for rgb2lab
    img1 = image1.astype(np.float32) / 255.0 if image1.dtype != np.float32 else image1
    img2 = image2.astype(np.float32) / 255.0 if image2.dtype != np.float32 else image2

    lab_image1 = rgb2lab(img1)
    lab_image2 = rgb2lab(img2)
    return deltaE_cie76(lab_image1, lab_image2)


def find_unwanted_images_by_color_distance(
    data_array: Sequence[np.ndarray],
    reference_unwanted_images: Sequence[np.ndarray],
    color_distance_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Given a set of images and a list of 'unwanted' reference images, remove all images
    whose color distance to ANY reference image is below a given threshold.

    Returns:
        cleaned_images: np.ndarray of kept images
        unwanted_images: np.ndarray of removed images
        unwanted_indices: list of indices (relative to data_array) that were removed
    """
    cleaned_data: List[np.ndarray] = []
    unwanted_images: List[np.ndarray] = []
    unwanted_indices: List[int] = []

    for idx, image in enumerate(data_array):
        similar = False
        for reference_image in reference_unwanted_images:
            color_distance = calculate_color_distance(image, reference_image)
            # 'all' here means every pixel is below the threshold
            if (color_distance < color_distance_threshold).all():
                similar = True
                break

        if similar:
            unwanted_images.append(image)
            unwanted_indices.append(idx)
        else:
            cleaned_data.append(image)

    # Convert lists to arrays (if not empty)
    cleaned_array = np.stack(cleaned_data, axis=0) if cleaned_data else np.empty((0,))
    unwanted_array = np.stack(unwanted_images, axis=0) if unwanted_images else np.empty((0,))

    return cleaned_array, unwanted_array, unwanted_indices


def cleaned_dataset(
    images: np.ndarray,
    color_distance_threshold: float,
    similarity_threshold: float = 0.62,
    reference_indices: Tuple[int, int] = (10, 33),
    explicit_reference_unwanted_images: Sequence[np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    High-level helper to:
        1. Detect 'odd' images by histogram difference from the first image.
        2. Choose two of those odd images as reference 'unwanted' samples (unless
           explicit_reference_unwanted_images is provided).
        3. Remove all images that are color-close to these references.

    Args:
        images: np.ndarray of shape (N, H, W, 3)
        color_distance_threshold: threshold for CIE76 distance
        similarity_threshold: histogram intersection threshold used for finding 'odd' images
        reference_indices: indices (within odd_images) used as reference samples
        explicit_reference_unwanted_images: optional pre-defined list of reference images.
            If provided, we skip automatic selection from odd_images.

    Returns:
        cleaned_images: np.ndarray with unwanted images removed
        unwanted_images: np.ndarray of images removed
        unwanted_indices: list of indices removed (relative to input 'images')
    """
    # 1) Build reference histogram from the first image (assumed 'normal')
    reference_image = images[0]
    reference_histogram = compute_histogram(reference_image)

    # 2) Find 'odd' images via histogram
    odd_images: List[np.ndarray] = []
    odd_indices: List[int] = []

    for idx, image in enumerate(images):
        if is_histogram_different(image, reference_histogram, similarity_threshold):
            odd_images.append(image)
            odd_indices.append(idx)

    # 3) Choose reference unwanted images
    if explicit_reference_unwanted_images is not None:
        reference_unwanted_images = list(explicit_reference_unwanted_images)
    else:
        reference_unwanted_images: List[np.ndarray] = []
        i1, i2 = reference_indices

        if (
            0 <= i1 < len(odd_images)
            and 0 <= i2 < len(odd_images)
        ):
            reference_unwanted_images.append(odd_images[i1])
            reference_unwanted_images.append(odd_images[i2])
        else:
            # Fallback: if something is wrong with indices, just use all odd images as reference
            reference_unwanted_images = odd_images.copy()

    # 4) Remove images similar to reference unwanted images
    cleaned_images, unwanted_images, unwanted_indices = find_unwanted_images_by_color_distance(
        images, reference_unwanted_images, color_distance_threshold
    )

    return cleaned_images, unwanted_images, unwanted_indices
