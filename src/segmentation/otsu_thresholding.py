import numpy as np

from skimage.filters import threshold_otsu

def apply_otsu_threshold(image):
    """
    Applies Otsu's thresholding method to segment an image.

    Parameters:
        image (numpy.ndarray): The input image array. Can have intensity values in any range and may contain NaNs.

    Returns:
        numpy.ndarray: The binary image after applying Otsu's thresholding, with pixel values set to 0 or 255 (uint8).
    """
    if image.max() <= 1.0:
        image = image * 255

    if np.isnan(image).any():
        nan_mean = np.nanmean(image)
        image = np.nan_to_num(image, nan=nan_mean)

    best_threshold = threshold_otsu(image)
    thresholded_image = np.where(image < best_threshold, 0, 255).astype(np.uint8)

    return thresholded_image