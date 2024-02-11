import numpy as np

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

    print("------------------------------")
    print(f"True Positives: {TP_percentage:.2f}%, False Positives: {FP_percentage:.2f}%, True Negatives: {TN_percentage:.2f}%, False Negatives: {FN_percentage:.2f}%")

    return TP, FP, TN, FN

def calculate_performance(TP, FP, TN, FN):
    # Ensure no division by zero
    epsilon = 1e-7
    total_cases = TP + FP + TN + FN

    accuracy = (TP + TN) / total_cases * 100
    sensitivity = TP / float(TP + FN + epsilon)
    specificity = TN / float(TN + FP + epsilon)
    auc = (sensitivity + specificity) / 2.0

    print("------------------------------")
    print(f"Accuracy: {accuracy:.2f}%")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("AUC:", auc)

    return accuracy, sensitivity, specificity, auc