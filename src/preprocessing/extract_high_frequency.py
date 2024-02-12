import numpy as np
from skimage.filters import gaussian

def high_freq(input_image):
    image = (255 * input_image).astype(np.uint8)

    filtered_img = image.astype(np.float16) / 255.0 - gaussian(image, sigma=3)
    filtered_img = filtered_img - np.min(filtered_img)
    output_image = (255 * filtered_img).astype(np.uint8)
    return output_image