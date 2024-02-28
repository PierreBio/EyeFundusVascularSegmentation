import numpy as np
from skimage import filters, exposure
from skimage.util import img_as_ubyte
import numpy as np

from PIL import Image

def apply_gaussian_filter(image, sigma=1):
    """
    Applique un filtre gaussien à l'image.

    :param image: Image d'entrée, numpy array 2D.
    :param sigma: Écart-type pour le filtre gaussien.
    :return: Image filtrée.
    """
    return filters.gaussian(image, sigma=sigma)

def apply_clahe(image, clip_limit=0.01, nbins=256):
    """
    Applique le CLAHE (Contrast Limited Adaptive Histogram Equalization) à l'image.

    :param image: Image d'entrée, numpy array 2D.
    :param clip_limit: Seuil pour le contraste.
    :param nbins: Nombre de bins pour l'histogramme.
    :return: Image après application du CLAHE.
    """
    image = img_as_ubyte(image)
    clahe = exposure.equalize_adapthist(image, clip_limit=clip_limit, nbins=nbins)
    return clahe

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