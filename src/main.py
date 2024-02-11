import numpy as np
import matplotlib.pyplot as plt
import cv2

from src.preprocessing.contrast import ImageContraster
from src.preprocessing.median_filter import median_filter
from src.segmentation.frangi_filter import frangi_vesselness_filter
from src.segmentation.convolution_filter import circular_averaging_filter
from src.segmentation.genetic import genetic
from src.morphology.operations import bridge_unconnected_pixels, closing_operation, diagonal_fill


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
    radius = 5
    filtered_image = circular_averaging_filter(image_frangi, radius)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Résultat après traitement')
    plt.axis('off')
    plt.show()

    # 5. Seuillage d'Otsu
    image_otsu_thresholded = apply_otsu_threshold(filtered_image)
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