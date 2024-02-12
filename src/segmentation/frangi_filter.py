import numpy as np
import math
from scipy import signal

def compute_hessian_2d(image, sigma):
    """
    Computes the Hessian matrix of an image at a given scale (sigma).

    Parameters:
    - image (numpy.ndarray): The input image.
    - sigma (float): The scale at which the Hessian matrix is computed.

    Returns:
    - (Dxx, Dxy, Dyy) (tuple): Second-order derivatives of the image.
    """
    image = np.array(image, dtype=float)
    sigma = np.array(sigma, dtype=float)
    kernel_radius = np.round(3 * sigma)

    [X, Y] = np.mgrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]

    factor = 1 / (2 * math.pi * sigma**4)
    exponential = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

    DGaussxx = factor * (X**2 / sigma**2 - 1) * exponential
    DGaussxy = factor / sigma**2 * (X * Y) * exponential
    DGaussyy = factor * (Y**2 / sigma**2 - 1) * exponential

    Dxx = signal.convolve2d(image, DGaussxx, boundary='fill', mode='same', fillvalue=0)
    Dxy = signal.convolve2d(image, DGaussxy, boundary='fill', mode='same', fillvalue=0)
    Dyy = signal.convolve2d(image, DGaussyy, boundary='fill', mode='same', fillvalue=0)

    return Dxx, Dxy, Dyy


def eigen_analysis_2d(Dxx, Dxy, Dyy):
    """
    Performs eigen analysis on the Hessian matrix components to find the principal
    directions and magnitudes of the local structures in the image.

    Parameters:
    - Dxx, Dxy, Dyy (numpy.ndarray): Components of the Hessian matrix.

    Returns:
    - (Lambda1, Lambda2, Ix, Iy) (tuple): The eigenvalues and eigenvectors of the Hessian matrix.
    """
    if len(Dxx.shape) != 2:
        return 0

    tmp = np.sqrt((Dxx - Dyy)**2 + 4*Dxy**2)

    eigenvector_x = 2*Dxy
    eigenvector_y = Dyy - Dxx + tmp

    magnitude = np.sqrt(eigenvector_x**2 + eigenvector_y**2)
    valid = magnitude != 0

    eigenvector_x[valid] = eigenvector_x[valid] / magnitude[valid]
    eigenvector_y[valid] = eigenvector_y[valid] / magnitude[valid]

    # Alternate eigenvector direction
    alt_eigenvector_x = -eigenvector_y
    alt_eigenvector_y = eigenvector_x

    # Eigenvalues
    eigenvalue1 = 0.5 * (Dxx + Dyy + tmp)
    eigenvalue2 = 0.5 * (Dxx + Dyy - tmp)

    # Swap if absolute value of first eigenvalue is larger than the second
    swap_condition = abs(eigenvalue1) > abs(eigenvalue2)

    Lambda1 = np.where(swap_condition, eigenvalue2, eigenvalue1)
    Lambda2 = np.where(swap_condition, eigenvalue1, eigenvalue2)

    Ix = np.where(swap_condition, eigenvector_x, alt_eigenvector_x)
    Iy = np.where(swap_condition, eigenvector_y, alt_eigenvector_y)

    return Lambda1, Lambda2, Ix, Iy


def frangi_vesselness_filter(image, options=None):
    """
    Enhances vessel-like structures in an image using the Frangi vesselness filter.

    Parameters:
    - image (numpy.ndarray): The input image.
    - options (dict, optional): Configuration options for the filter.

    Returns:
    - outIm (numpy.ndarray): The vessel-enhanced image.
    """
    image = np.array(image, dtype=float)

    # Default filter options
    default_options = {
        'FrangiScaleRange': (1, 10),  # Sigma range
        'FrangiScaleRatio': 2,        # Sigma step size
        'FrangiBetaOne': 0.5,         # Frangi correction constant
        'FrangiBetaTwo': 15,          # Frangi correction constant
        'verbose': True,              # Show progress
        'BlackWhite': True,           # Detect black ridges
    }
    if options:
        for key, value in options.items():
            default_options[key] = value

    sigmas = np.arange(default_options['FrangiScaleRange'][0], default_options['FrangiScaleRange'][1], default_options['FrangiScaleRatio'])
    sigmas.sort()  # Ensure ascending order

    beta = 2 * default_options['FrangiBetaOne']**2
    c = 2 * default_options['FrangiBetaTwo']**2

    vesselness_image = np.zeros_like(image, dtype=float)
    vesselness_scale = np.zeros(image.shape + (len(sigmas),))
    for i, sigma in enumerate(sigmas):
        
        if default_options['verbose']:
            print(f'Processing scale: {sigma}')

        Dxx, Dxy, Dyy = compute_hessian_2d(image, sigma)
        Dxx *= sigma**2
        Dxy *= sigma**2
        Dyy *= sigma**2

        Lambda2, Lambda1, Ix, Iy = eigen_analysis_2d(Dxx, Dxy, Dyy)

        Rb = (Lambda2 / Lambda1)**2
        S2 = Lambda1**2 + Lambda2**2

        Ifiltered = np.exp(-Rb / beta) * (1 - np.exp(-S2 / c))

        if default_options['BlackWhite']:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        vesselness_scale[:, :, i] = Ifiltered

    if len(sigmas) > 1:
        outIm = np.max(vesselness_scale, axis=2)
    else:
        outIm = vesselness_scale.reshape(image.shape)

    return outIm