import numpy as np
import cv2
import pandas as pd

from src.preprocessing.contrast import enhance_contrast
from src.preprocessing.median_filter import median_filter
from src.preprocessing.other_median_filter import median_filter_disk
from src.preprocessing.equalize import my_equalize
from src.preprocessing.extract_high_frequency import high_freq
from src.segmentation.frangi_filter import frangi_vesselness_filter
from src.segmentation.convolution_filter import circular_averaging_filter, fir_filter_image
from src.segmentation.otsu_thresholding import apply_otsu_threshold
from src.morphology.erosion import erode
from src.morphology.operations import bridge_unconnected_pixels, closing_operation, diagonal_fill
from src.performance.analysis import my_segmentation, total_evaluation


def process_image_pierre(gray_image):
    image_clahe = enhance_contrast(gray_image, blocks = 16, threshold = 8.0)
    image_median_filtered = median_filter(image_clahe, filter_size=3)

    custom_options = {
        'FrangiScaleRange': (0.1, 2),
        'FrangiScaleRatio': 0.05,
        'FrangiBetaOne': 0.3,
        'FrangiBetaTwo': 15,
        'verbose': False,
        'BlackWhite': True
    }

    image_frangi = frangi_vesselness_filter(image_median_filtered, custom_options)
    filtered_image_conv = circular_averaging_filter(image_frangi, 3)
    filtered_image_fir = fir_filter_image(filtered_image_conv, np.array([0.01, 0.1, 0.1, 0.05, 0.01]))
    image_otsu_thresholded = apply_otsu_threshold(filtered_image_fir)

    binary_closed = closing_operation(image_otsu_thresholded, structure=np.ones((2,2)))
    binary_diagonal_filled = diagonal_fill(binary_closed)
    binary_bridged = bridge_unconnected_pixels(binary_diagonal_filled)

    return binary_bridged

def process_image_pascal(image):
    filtered_img = median_filter_disk(image, 3)
    locally_equalized_img = my_equalize(filtered_img)
    hi_freq_img = high_freq(locally_equalized_img)
    img_eros = erode(hi_freq_img)
    filtered_eros = median_filter_disk(img_eros, 2)

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

    result_list = []
    image_folder = "data/images_IOSTAR/"
    for item in all_images:
        print("processing image: {}".format(item["img_path"]))
        image_path = image_folder + item["img_path"]
        ground_truth_path = image_folder + item["ground_truth_path"]

        current_result = process_img(image_path, ground_truth_path)
        result_list.append(current_result)

    result_DF = pd.DataFrame(result_list)
    print(result_DF)
