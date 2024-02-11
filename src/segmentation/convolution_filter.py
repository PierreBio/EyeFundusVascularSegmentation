import numpy as np
from scipy.ndimage import convolve

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

def circular_averaging_filter(image, radius):
    h, w = image.shape
    mask = create_circular_mask(h, w, radius=radius)
    mask = mask.astype(float) / mask.sum()  # Normalize the mask

    # Apply the filter using 2D convolution
    filtered_image = convolve(image, mask, mode='constant', cval=0.0)

    return filtered_image