< Back to [README](../README.md)

# Method

A. Preprocessing

The preprocessing stage aims to enhance the quality of retinal images for better segmentation results. It includes:

- CLAHE (Contrast-Limited Adaptive Histogram Equalization): Applied to the grey image to enhance color contrast and improve visibility of details.
- Median Filtering: Used to reduce noise in the image, smoothing it without blurring the edges of vessels.

B. Segmentation

Segmentation is the core process, involving several steps to accurately identify and delineate blood vessels:

- **Frangi Filter**: Utilizes the Hessian matrix to enhance the visibility of blood vessels, optimizing the response for varying vessel widths.
- **Convolution Filtering**: A 2D convolution process, possibly integrated with a 2D FIR (Finite Impulse Response) filter like a Circular Averaging filter, is used to further refine vessel visibility.
- **Otsu's Thresholding**: An automatic method to find the optimal threshold for segmenting the vessels from the background, based on minimizing within-class variance.

C. Morphological Processing

Following segmentation, morphological operations are employed to refine the segmentation results:

- Closing Operation: Aims to close small holes and gaps in the segmented vessels.
- Diagonal Fill: Eliminates background noise by filling in gaps in the vessel structure.
- Bridging Unconnected Pixels: Connects isolated vessel pixels to ensure continuous vessel representation.

D. Performance Analysis

The effectiveness of the segmentation method is evaluated using a confusion matrix, focusing on:

- Sensitivity (True Positive Rate): Measures the proportion of actual positives correctly identified.
- Specificity (True Negative Rate): Measures the proportion of actual negatives correctly identified.
- Accuracy: The overall correctness of the segmentation.
- Area Under the Curve (AUC): Evaluates the trade-off between sensitivity and specificity across different thresholds.
