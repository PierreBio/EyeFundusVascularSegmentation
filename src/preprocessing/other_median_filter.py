from skimage.morphology import disk
from skimage.filters.rank import median

def median_filter_disk(input_image, n):
    output_image = median(input_image, disk(n))
    return output_image