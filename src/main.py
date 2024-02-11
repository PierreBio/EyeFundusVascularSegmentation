import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion
import cv2
from PIL import Image

from src.preprocessing.contrast import ImageContraster
from src.segmentation.frangi_filter import frangi_vesselness_filter
from src.segmentation.genetic import genetic
from src.segmentation.otsu import fast_otsu

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


def convolution_filter(image, kernel):
    """
    Apply a convolution filter to an image.
    This function uses 'reflect' as the boundary condition and 'valid' as the convolution mode,
    meaning the output image will be smaller than the input image if the kernel size is larger than 1x1.
    """
    return signal.convolve2d(image, kernel, boundary='symm', mode='same')

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

def median_filter(input_image, filter_size):
    # Convert PIL Image to numpy array if it is not already an array
    if isinstance(input_image, Image.Image):
        input_image = np.array(input_image)

    temp_image = np.pad(input_image, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)), 'reflect')
    output_image = np.zeros_like(input_image)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            window = temp_image[i:i+filter_size, j:j+filter_size]
            median = np.median(window)
            output_image[i, j] = median

    return output_image


def apply_otsu_threshold(image):
    if image.max() <= 1.0:
        image = image * 255

    genrtic = genetic(image)
    best_threshold = genrtic.get_threshold()

    print(f"Best threshold: {best_threshold}")

    # Plot the histogram of the image
    plt.hist(image.ravel(), bins=256, range=(0, 256))
    plt.title("Histogram")
    plt.show()

    # Use the threshold to create a binary image
    thresholded_image = np.where(image < best_threshold, 0, 255).astype(np.uint8)

    # Check the unique values in the thresholded image
    print(f"Unique values in the thresholded image: {np.unique(thresholded_image)}")

    # Show the thresholded image
    plt.imshow(thresholded_image, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis('off')
    plt.show()


    return thresholded_image

if __name__ == "__main__":
    # Charger l'image et la convertir en niveaux de gris si nécessaire
    image_path = 'data/images_IOSTAR/star01_OSC.jpg'
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. Appliquer la CLAHE
    # CLAHE
    # contraster
    icter = ImageContraster()
    image_clahe = icter.enhance_contrast(gray_image, method = "CLAHE", blocks = 8, threshold = 10.0)

    # 2. Filtrage médian
    image_median_filtered = median_filter(image_clahe, filter_size=3)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Create a figure with two subplots

    # Display the CLAHE result
    axs[0].imshow(image_clahe, cmap='gray')
    axs[0].set_title('After CLAHE')
    axs[0].axis('off')  # Hide axes for better visualization

    # Display the result after median filtering
    axs[1].imshow(image_median_filtered, cmap='gray')
    axs[1].set_title('After Median Filtering')
    axs[1].axis('off')  # Hide axes for better visualization

    plt.tight_layout()  # Adjust the layout to make sure there's no overlap
    plt.show()

    # 3. Appliquer le filtre de Frangi
    custom_options = {
        'FrangiScaleRange': (1, 3.5),  # The range of sigmas starting from a value greater than 0
        'FrangiScaleRatio': 0.5,       # The step size between consecutive sigmas
        # Other parameters can remain unchanged or be customized further
        'FrangiBetaOne': 0.5,
        'FrangiBetaTwo': 15,
        'verbose': True,
        'BlackWhite': True
    }

    # Pass the custom options when you call FrangiFilter2D
    image_frangi = frangi_vesselness_filter(image_median_filtered, custom_options)
    plt.imshow(image_frangi, cmap='gray')
    plt.title('Résultat après traitement')
    plt.axis('off')
    plt.show()

    # 4. Filtre de convolution

    # 5. Seuillage d'Otsu
    image_otsu_thresholded = apply_otsu_threshold(image_frangi)
    plt.imshow(image_otsu_thresholded, cmap='gray')
    plt.title('Résultat après traitement')
    plt.axis('off')
    plt.show()

    # 6. Opérations morphologiques
    # Fermeture
    binary_closed = closing_operation(image_otsu_thresholded, structure=np.ones((3,3)))
    # Remplissage diagonal
    binary_diagonal_filled = diagonal_fill(binary_closed)
    # Pontage des pixels non connectés
    binary_bridged = bridge_unconnected_pixels(binary_diagonal_filled)

    # Visualiser le résultat final
    plt.imshow(binary_bridged, cmap='gray')
    plt.title('Résultat après traitement')
    plt.axis('off')
    plt.show()

    # 7. Analyse de performance