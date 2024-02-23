import numpy as np
from scipy.ndimage import convolve
from scipy.signal import convolve2d

def circular_averaging_filter(image, radius):
    """
    Applies a circular averaging filter to an image.

    Parameters:
        image (numpy.ndarray): The input image to be filtered.
        radius (int): The radius of the circular averaging kernel.

    Returns:
        numpy.ndarray: The image after applying the circular averaging filter.
    """
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    kernel = x**2 + y**2 <= radius**2
    kernel = kernel.astype(float) / kernel.sum()
    return convolve(image, kernel, mode='reflect')

def fir_filter_image(image, fir_coeff):
    """
    Applies a Finite Impulse Response (FIR) filter to an image.

    Parameters:
        image (numpy.ndarray): The input image to be filtered.
        fir_coeff (numpy.ndarray): The coefficients of the FIR filter.

    Returns:
        numpy.ndarray: The image after applying the FIR filter.
    """
    fir_filtered_rows = convolve2d(image, fir_coeff.reshape(1, -1), mode='same')
    return convolve2d(fir_filtered_rows, fir_coeff.reshape(-1, 1), mode='same')
