import numpy as np
from PIL import Image


def median_filter(input_image, filter_size):
    """
    Apply a median filter to an image to reduce noise.

    Parameters:
    - input_image (numpy.ndarray or PIL.Image.Image): The image to be filtered.
    - filter_size (int): The size of the median filter window, must be an odd integer.

    Returns:
    - numpy.ndarray: The median-filtered image.

    The function supports both numpy arrays and PIL image inputs, converting the latter to numpy arrays. It pads the image using reflection to handle borders.
    """
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)

    temp_image = np.pad(input_image, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), 'reflect')
    output_image = np.zeros_like(input_image)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            window = temp_image[i:i+filter_size, j:j+filter_size]
            median = np.median(window)
            output_image[i, j] = median

    return output_image