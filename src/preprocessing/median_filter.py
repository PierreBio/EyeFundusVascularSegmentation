import numpy as np
from PIL import Image


def median_filter(input_image, filter_size):
    # Convert PIL Image to numpy array if it is not already an array
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