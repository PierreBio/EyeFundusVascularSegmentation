
from skimage.morphology import erosion, disk


def erode(binary_image):
    """
    Applies morphological erosion to an image.
    """
    output_image = erosion(binary_image, disk(2))
    return output_image