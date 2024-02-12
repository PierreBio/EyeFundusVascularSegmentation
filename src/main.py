import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from src.preprocessing.contrast import ImageContraster
from src.preprocessing.median_filter import median_filter
from src.preprocessing.other_median_filter import median_filter_disk
from src.preprocessing.equalize import my_equalize
from src.preprocessing.extract_high_frequency import high_freq
from src.segmentation.frangi_filter import frangi_vesselness_filter
from src.segmentation.convolution_filter import circular_averaging_filter
from src.segmentation.genetic import genetic
from src.morphology.erosion import erode
from src.morphology.operations import bridge_unconnected_pixels, closing_operation, diagonal_fill
from src.performance.analysis import my_segmentation, total_evaluation, SHOW_IMAGES, SHOW_NUMBERS

def apply_otsu_threshold(image):
    if image.max() <= 1.0:
        image = image * 255

    genrtic = genetic(image)
    best_threshold = genrtic.get_threshold()

    if SHOW_NUMBERS:
        print(f"Best threshold: {best_threshold}")

    # Plot the histogram of the image
    if SHOW_IMAGES:
        plt.hist(image.ravel(), bins=256, range=(0, 256))
        plt.title("Histogram")
        plt.show()

    # Use the threshold to create a binary image
    thresholded_image = np.where(image < 150, 0, 255).astype(np.uint8)

    # Check the unique values in the thresholded image
    if SHOW_NUMBERS:
        print(f"Unique values in the thresholded image: {np.unique(thresholded_image)}")

    # Show the thresholded image
    if SHOW_IMAGES:
        plt.imshow(thresholded_image, cmap='gray')
        plt.title("Thresholded Image")
        plt.axis('off')
        plt.show()

    return thresholded_image

def process_image_pierre(gray_image):
    # 1. Appliquer la CLAHE
    # CLAHE
    # contraster
    icter = ImageContraster()
    image_clahe = icter.enhance_contrast(gray_image, method = "CLAHE", blocks = 8, threshold = 10.0)

    # 2. Filtrage médian
    image_median_filtered = median_filter(image_clahe, filter_size=3)

    if SHOW_IMAGES:
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
        'FrangiScaleRange': (0.25, 2),  # Plus petites échelles
        'FrangiScaleRatio': 0.25,      # Pas plus fin
        'FrangiBetaOne': 0.25,         # Sensibilité accrue
        'FrangiBetaTwo': 10,           # Diminution de la correction pour le fond
        'verbose': False,
        'BlackWhite': True
    }

    # Pass the custom options when you call FrangiFilter2D
    image_frangi = frangi_vesselness_filter(image_median_filtered, custom_options)
    if SHOW_IMAGES:
        plt.imshow(image_frangi, cmap='gray')
        plt.title('Résultat après traitement')
        plt.axis('off')
        plt.show()

    # 4. Filtre de convolution
    #radius = 5
    #filtered_image = circular_averaging_filter(image_frangi, radius)
    # if SHOW_IMAGES:
    #     plt.imshow(filtered_image, cmap='gray')
    #     plt.title('Résultat après traitement')
    #     plt.axis('off')
    #     plt.show()

    # 5. Seuillage d'Otsu
    image_otsu_thresholded = apply_otsu_threshold(image_frangi)
    if SHOW_IMAGES:
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
    if SHOW_IMAGES:
        plt.imshow(binary_bridged, cmap='gray')
        plt.axis('off')
        plt.show()

    return binary_bridged

def process_image_pascal(image):
    # version de Pascal
    filtered_img = median_filter_disk(image, 3)
    locally_equalized_img = my_equalize(filtered_img)
    hi_freq_img = high_freq(locally_equalized_img)
    img_eros = erode(hi_freq_img)
    filtered_eros = median_filter_disk(img_eros, 2)

    # plt.imshow(filtered_eros, cmap='gray')
    # plt.show()

    return filtered_eros



def process_img(image_path, ground_truth_path):

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ground_truth = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    
    image_out_pierre = process_image_pierre(gray_image)
    image_out_pascal = process_image_pascal(gray_image)

    segmented_img_pierre = my_segmentation(image_out_pierre)
    segmented_img_pascal = my_segmentation(image_out_pascal)

    ACCU_pi, RECALL_pi, TP_pi, FP_pi, TN_pi, FN_pi = total_evaluation(gray_image, segmented_img_pierre, ground_truth, "Pierre")
    ACCU_pa, RECALL_pa, TP_pa, FP_pa, TN_pa, FN_pa = total_evaluation(gray_image, segmented_img_pascal, ground_truth, "Pascal")

    results = {"ACCU_pi": ACCU_pi, "RECALL_pi": RECALL_pi, "TP_pi": TP_pi, "FP_pi": FP_pi, "TN_pi": TN_pi, "FN_pi": FN_pi, "ACCU_pa": ACCU_pa, "RECALL_pa": RECALL_pa, "TP_pa": TP_pa, "FP_pa": FP_pa, "TN_pa": TN_pa, "FN_pa": FN_pa}
    return results

if __name__ == "__main__":

    all_images = [
        {"img_path": "star01_OSC.jpg", "ground_truth_path": "GT_01.png"},
        {"img_path": "star02_OSC.jpg", "ground_truth_path": "GT_02.png"},
        {"img_path": "star03_OSN.jpg", "ground_truth_path": "GT_03.png"},
        {"img_path": "star08_OSN.jpg", "ground_truth_path": "GT_08.png"},
        {"img_path": "star21_OSC.jpg", "ground_truth_path": "GT_21.png"},
        {"img_path": "star26_ODC.jpg", "ground_truth_path": "GT_26.png"},
        {"img_path": "star28_ODN.jpg", "ground_truth_path": "GT_28.png"},
        {"img_path": "star32_ODC.jpg", "ground_truth_path": "GT_32.png"},
        {"img_path": "star37_ODN.jpg", "ground_truth_path": "GT_37.png"},
        {"img_path": "star48_OSN.jpg", "ground_truth_path": "GT_48.png"},
    ]

    # Charger l'image et la convertir en niveaux de gris si nécessaire
    
    result_list = []
    image_folder = "data/images_IOSTAR/"
    for item in all_images:
        # dirty way to build path, should use os.separator or something
        print("processing image: {}".format(item["img_path"]))
        image_path = image_folder + item["img_path"]
        ground_truth_path = image_folder + item["ground_truth_path"]

        current_result = process_img(image_path, ground_truth_path)
        result_list.append(current_result)

    # image_path = 'data/images_IOSTAR/star03_OSN.jpg'
    # ground_truth_path = 'data/images_IOSTAR/GT_03.png'
    # current_result = process_img(image_path, ground_truth_path)
    # result_list.append(current_result)

    result_DF = pd.DataFrame(result_list)
    print(result_DF)
    
    