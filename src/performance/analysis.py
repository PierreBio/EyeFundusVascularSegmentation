import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import thin

SHOW_IMAGES = False
SHOW_NUMBERS = False

def compare_images(ground_truth, predicted):
    predicted = (predicted > 0).astype(np.uint8)
    ground_truth = (ground_truth > 0).astype(np.uint8)

    TP = np.sum((predicted == 1) & (ground_truth == 1))
    TN = np.sum((predicted == 0) & (ground_truth == 0))
    FP = np.sum((predicted == 1) & (ground_truth == 0))
    FN = np.sum((predicted == 0) & (ground_truth == 1))

    total_pixels = TP + FP + TN + FN
    TP_percentage = (TP / total_pixels) * 100
    FP_percentage = (FP / total_pixels) * 100
    TN_percentage = (TN / total_pixels) * 100
    FN_percentage = (FN / total_pixels) * 100

    if SHOW_NUMBERS:
        print("------------------------------")
        print(f"True Positives: {TP_percentage:.2f}%, False Positives: {FP_percentage:.2f}%, True Negatives: {TN_percentage:.2f}%, False Negatives: {FN_percentage:.2f}%")

    return TP_percentage, FP_percentage, TN_percentage, FN_percentage

def calculate_performance(TP, FP, TN, FN):
    # Ensure no division by zero
    epsilon = 1e-7
    total_cases = TP + FP + TN + FN

    accuracy = (TP + TN) / total_cases * 100
    sensitivity = TP / float(TP + FN + epsilon)
    specificity = TN / float(TN + FP + epsilon)
    auc = (sensitivity + specificity) / 2.0

    if SHOW_NUMBERS:
        print("------------------------------")
        print(f"Accuracy: {accuracy:.2f}%")
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("AUC:", auc)

    return accuracy, sensitivity, specificity, auc

def my_segmentation(processed_image):
    """
    Greatly inspired by my_segmentation in tp2_script.py
    Only here the image processing is done before.
    """
    # creating mask
    nrows, ncols = processed_image.shape
    row, col = np.ogrid[:nrows, :ncols]
    img_mask = (np.ones(processed_image.shape)).astype(np.bool_)
    invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
    img_mask[invalid_pixels] = 0

    # convert back binary image to greyscale if needed
    # (quick & dirty heuristic: if mean < 1 -> very few values above 1 => binary image
    if np.mean(processed_image) < 1.0:
        processed_image = 255.0 - processed_image * 255.0

    image_out = (img_mask & (processed_image < 80))
    return image_out

def evaluate(img_out, img_GT):
    """
    Pulled from tp2_script.py, written by Gianni Franchi.
    Computes the accuracy and recall for an image, given the ground truth
    """
    GT_skel = thin(img_GT, max_num_iter = 15) # On suppose que la demie epaisseur maximum
    img_out_skel = thin(img_out, max_num_iter = 15) # d'un vaisseau est de 15 pixels...
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

def show_last_img(img_src, img_out, img_out_skel, img_GT, GT_skel):
    """
    Pulled from tp2_script.py, written by Gianni Franchi.
    Shows the source image, the final processed image, the skeleton, the ground truth and the ground truth skeleton.
    """
    if SHOW_IMAGES:
        plt.subplot(231)
        plt.imshow(img_src, cmap = 'gray')
        plt.title('Image Originale')
        plt.subplot(232)
        plt.imshow(img_out)
        plt.title('Segmentation')
        plt.subplot(233)
        plt.imshow(img_out_skel)
        plt.title('Segmentation squelette')
        plt.subplot(235)
        plt.imshow(img_GT)
        plt.title('Vérité Terrain')
        plt.subplot(236)
        plt.imshow(GT_skel)
        plt.title('Vérité Terrain Squelette')
        plt.show()

def total_evaluation(image_src, processed_image, ground_truth, method_name):
    # 7. Analyse de performance
    TP, FP, TN, FN = compare_images(ground_truth, processed_image)
    calculate_performance(TP, FP, TN, FN)

    ACCU, RECALL, img_out_skel, GT_skel = evaluate(processed_image, ground_truth)
    show_last_img(image_src, processed_image, img_out_skel, ground_truth, GT_skel)
    if SHOW_NUMBERS:
        print(f'{method_name}\'s version = Accuracy = {ACCU*100:.3f} % - Recall = {RECALL*100:.3f} %')

    return ACCU, RECALL, TP, FP, TN, FN