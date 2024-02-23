import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

def bridge_unconnected_pixels(binary_image):
    """
    Bridges unconnected pixels in a binary image to enhance connectivity.

    This function specifically targets diagonal disconnections, connecting pixels if their diagonal neighbors are set but they themselves are not.

    Parameters:
    - binary_image (numpy.ndarray): The binary image to be processed.

    Returns:
    - numpy.ndarray: The image with bridged unconnected pixels.
    """
    bridged_image = binary_image.copy()
    rows, cols = binary_image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            neighborhood = binary_image[i-1:i+2, j-1:j+2]

            if neighborhood[0, 0] and neighborhood[2, 2] and not neighborhood[1, 1]:
                bridged_image[i, j] = 1
            elif neighborhood[2, 0] and neighborhood[0, 2] and not neighborhood[1, 1]:
                bridged_image[i, j] = 1

    return bridged_image

def closing_operation(binary_image, structure=np.ones((3,3))):
    """
    Applies a morphological closing operation to a binary image.

    Closing is performed using a specified structuring element, which dilates the image and then erodes it.

    Parameters:
    - binary_image (numpy.ndarray): The binary image to process.
    - structure (numpy.ndarray): Structuring element for the closing operation.

    Returns:
    - numpy.ndarray: The image after the closing operation.
    """
    dilated_image = binary_dilation(binary_image, structure=structure)
    closed_image = binary_erosion(dilated_image, structure=structure)
    return closed_image

def diagonal_fill(binary_image):
    """
    Fills diagonal gaps in a binary image to improve object connectivity.

    This function checks for diagonal gaps between pixels and fills them to create a continuous object.

    Parameters:
    - binary_image (numpy.ndarray): The binary image to process.

    Returns:
    - numpy.ndarray: The image with filled diagonal gaps.
    """
    filled_image = binary_image.copy()
    rows, cols = binary_image.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if binary_image[i-1, j-1] and binary_image[i+1, j+1] and not binary_image[i, j]:
                filled_image[i, j] = 1
            elif binary_image[i+1, j-1] and binary_image[i-1, j+1] and not binary_image[i, j]:
                filled_image[i, j] = 1

    return filled_image