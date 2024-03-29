import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, remove_small_objects, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from skimage.filters import frangi
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
import math
from skimage import data, filters
from matplotlib import pyplot as plt

from src.preprocessing.contrast import enhance_contrast
from src.preprocessing.median_filter import median_filter
from src.segmentation.convolution_filter import circular_averaging_filter, fir_filter_image
from src.segmentation.otsu_thresholding import apply_otsu_threshold
from src.morphology.operations import prune_small_branches

def my_segmentation(img, img_mask):
    image_clahe = enhance_contrast(img, blocks=14, threshold=8.0)
    image_median_filtered = median_filter(image_clahe, filter_size=3)

    frangi_params = {
        'scale_range': (2, 15),
        'scale_step': 1,
        'alpha': 1,
        'beta': 1,
        'gamma': 4,
        'black_ridges': True,
        'mode': 'wrap',
        'cval': 5
    }

    filtered_image = frangi(image_median_filtered, **frangi_params)
    filtered_image_conv = circular_averaging_filter(filtered_image, 2)
    filtered_image_fir = fir_filter_image(filtered_image_conv, np.array([0.01, 3, 3, 3, 0.01]))
    image_otsu_thresholded = apply_otsu_threshold(filtered_image_fir)

    binary_filtered = remove_small_objects(image_otsu_thresholded.astype(bool), min_size=250)
    binary_pruned_image = prune_small_branches(binary_filtered)

    img_out = (img_mask & binary_pruned_image).astype(np.uint8)
    return img_out

def evaluate(img_out, img_GT):
    GT_skel  = thin(img_GT, max_num_iter = 15) # On suppose que la demie epaisseur maximum
    img_out_skel  = thin(img_out, max_num_iter = 15) # d'un vaisseau est de 15 pixels...
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./data/images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./data/images_IOSTAR/GT_01.png')).astype(np.bool_)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out)
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel)
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT)
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel)
plt.title('Verite Terrain Squelette')
plt.show()
