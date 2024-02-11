import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


def bridge_unconnected_pixels(binary_image):
    """
    Connects nearby unconnected pixels (diagonal connection) in a binary image.
    """
    bridged_image = binary_image.copy()
    rows, cols = binary_image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Define the neighborhood
            neighborhood = binary_image[i-1:i+2, j-1:j+2]

            # Check the connectivity conditions
            # For example, if the upper left and lower right pixels are foreground, but the center is not
            if neighborhood[0, 0] and neighborhood[2, 2] and not neighborhood[1, 1]:
                bridged_image[i, j] = 1
            # Similar check for the other diagonal
            elif neighborhood[2, 0] and neighborhood[0, 2] and not neighborhood[1, 1]:
                bridged_image[i, j] = 1
            # Additional conditions can be added to bridge in other scenarios

    return bridged_image


def closing_operation(binary_image, structure=np.ones((3,3))):
    """
    Perform a closing operation on a binary image.

    :param binary_image: A binary (black and white) image.
    :param structure: The structuring element used for closing.
    :return: The image after applying the closing operation.
    """
    # Dilate then erode the image.
    dilated_image = binary_dilation(binary_image, structure=structure)
    closed_image = binary_erosion(dilated_image, structure=structure)
    return closed_image


def diagonal_fill(binary_image):
    """
    Fill diagonal gaps in a binary image.
    """
    filled_image = binary_image.copy()
    rows, cols = binary_image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            # Check for diagonal connectivity:
            # If there's a diagonal gap, fill it
            if binary_image[i-1, j-1] and binary_image[i+1, j+1] and not binary_image[i, j]:
                filled_image[i, j] = 1
            elif binary_image[i+1, j-1] and binary_image[i-1, j+1] and not binary_image[i, j]:
                filled_image[i, j] = 1

    return filled_image