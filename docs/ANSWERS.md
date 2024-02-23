< Back to [README](../README.md)

# Answers to questions

### Answer 1
- **Can you realize an optimized algorithm to segment the vascular network without using machine learning, nor deep learning?**

We are inspired from [Wiharto, 2019](https://thesai.org/Downloads/Volume10No6/Paper_54-Blood_Vessels_Segmentation.pdf). This article proposes a [method](docs/METHOD.md) to segment an image of retina. We used it in order to get similar results but it appears that **we didn't achieve to get the same performances** especially because we don't use exactly the same functions (Custom CLAHE, Frangi, Convolution, etc.) and parameters foreach step of the method.

__Note 1: They don't use the same performance values.__

For **images_IOSTAR/star01_OSC.jpg** we get the next results:
Accuracy = 0.734845443596949 , Recall = 0.7239470041526597

### Answer 2
- **Can you explain the evaluation function provided in the "main" Python script?**

This function is used to calculate performances of our algorithm.

Firstly, this function "normalizes" our ground truth image and our generated output image using skeletonization.

Then it compares the two images using their respective pixel color values:

- True Positive pixels (TP) are pixels found as white and really white.
- False Positive pixels (FP) are pixels found as white but black in reality.
- False Negative pixels (FN) are pixels found as black but white in reality.

It allows us finally to calculate the global "accuracy" and "recall" of our algorithm.

- The "accuracy" is the percent of "True Positive" pixels among all the pixels perceived as white by our algorithm (TP + FP).
- The "recall" is the percent of "True Positive" pixels among all the pixels which are really white (TP + FN).

__Note 1: Other values exist, like we can see in the Wiharto method.__

__Note 2: It's important to have balanced values between accuracy and recall. Unbalanced results can reflext an inefficient algorithm.__

### Answer 3
- **Why do we use two metrics (Precision and Recall)?**

We think we use these two metrics because we want to focus mainly on **the capacity of our algorithm to find blood vessels correspondance**. We don't focus on the capacity of finding the background pixels. It's explainable by the reason that we can find really easily a huge correspondance of True Negative pixels for example.

I  think that the following calculation can give a global idea of the performance of our algorithm:

- ("Accuracy" + "Recall") / 2

Note 1: If the result is close to 0.5, it means that our algorithm is not better that randomly choose black or white value.

__Other methods can choose to focus on other metrics. For example, in the Wiharto article, they calculate "Sensitivity", "Specificity", "Accuracy" and "Area under the curve". These metrics can have the same name as ours but don't reflect the same values at the end.__

### Answer 4
- **What role does skeletonization play in this evaluation function?**

As mentioned in the "Answer 2" part, skletonization is used to "normalize" our output image as such as our ground truth image. It means that we want to focus on the fact that our algorithm managed to find **vessels and networks, whatever their thickness** (which is constrained with a diameter of 15 pixels) in skeletonization function.

This skeletonization function **is consistent with the idea reflected by the use of the two metrics explained in Answer 3**: we focus on blood vessels correspondance and especially on their length more than their width because **width can be subjected to a lot of different variations depending parameters of our segmentation functions**. The more important is the vascular network.