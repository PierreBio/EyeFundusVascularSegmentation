from skimage.filters.rank import equalize
from skimage.morphology import disk

def my_equalize(input_image):
    output_image = equalize(input_image, disk(20))

    return output_image