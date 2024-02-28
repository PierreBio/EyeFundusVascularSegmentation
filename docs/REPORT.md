\documentclass[12pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}

\title{Report on Practical Work No. 2: Segmentation of Retinal Vascular Networks}
\author{Pierre Billaud – Pascal Mahé}
\date{}

\begin{document}

\maketitle

\tableofcontents

\section{Introduction}
This report aims to present the work carried out for Practical Work No. 2 by Pierre Billaud and Pascal Mahé.

\section{Responses to Questions}
\subsection{Can you develop an optimized algorithm to segment a vascular network without machine learning or deep learning?}
We were inspired by the article available here: Wiharto, 2019. The description of the method indicated in the article and our results constitute parts 2 to 5 of this report.

\subsection{Explain the evaluation function provided in the Python script}
The function calculates the performance of our algorithm.

\begin{verbatim}
def evaluate(img_out, img_GT):
    GT_skel  = thin(img_GT, max_num_iter = 15) # Assuming the maximum half
    img_out_skel  = thin(img_out, max_num_iter = 15) # thickness of a vessel is 15 pixels...
    TP = np.sum(img_out_skel & img_GT) # True Positives
    FP = np.sum(img_out_skel & ~img_GT) # False Positives
    FN = np.sum(GT_skel & ~img_out) # False Negatives
    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Recall
    return ACCU, RECALL, img_out_skel, GT_skel
\end{verbatim}

It starts by normalizing the ground truth image and our result by skeletonizing them. Then, the two images are compared based on the respective pixel values:
\begin{itemize}
    \item Pixels that are white in both images are true positives (TP)
    \item Pixels that are white in the generated image but black in reality are false positives (FP)
    \item Pixels that are black in the generated image but white in reality are false negatives (FN)
\end{itemize}
Finally, the precision and recall of our algorithm are calculated:
\begin{itemize}
    \item Precision is the rate of true positives among the white pixels of the generated image: \(ACCU = \frac{TP}{TP + FP}\)
    \item Recall is the rate of true positives among all the white pixels in the original image: \(RECALL = \frac{TP}{TP + FN}\)
\end{itemize}
It is important to have balanced precision and recall values; otherwise, it may indicate that the algorithm is not performing properly.

\subsection{Why do we use two metrics (Precision and Recall)?}
Two metrics are used because we want to differentiate blood vessels from the background. With only one of these metrics, it would be easy to overlook a deficient algorithm. The following calculation seems to provide a better idea of the overall performance of the algorithm:
\[(Accuracy + Recall) / 2\]

The article provides other metrics to measure its performance: specificity and the "area under the curve" (AUC) calculated as follows:
\[Specificity = \frac{TN}{TN + FP}\]

\[AUC = \frac{(Recall + Specificity)}{2}\]
(Note: in the article, recall is named sensitivity, but the calculation method is the same.)

\subsection{What role does skeletonization play in this evaluation function?}
Skeletonization is used to normalize the generated images compared to the ground truth. This allows us to focus the output of the algorithm on segmenting blood vessels, regardless of their thickness (bounded to 15 pixels in the skeletonization function). It brings us back to the original problem: generating a map of the vascular network. Skeletonization maps the different possible gray levels in the output to a binary response: whether a pixel is part of a blood vessel or not.

\section{Recall of the Problem}
The objective of the practical work is to extract an image of the retinal vascular network from laser scanning ophthalmoscopy images, a high-resolution retinal imaging technique (between 10 and 100 $\mu$m) and wide field – allowing most of the retina to be observed in a single image.
The analysis of the obtained fundus image allows for the diagnosis of several diseases, including high blood pressure, renal insufficiency, and various retinal diseases. The diagnosis requires precise segmentation of the network.
This segmentation, performed by human experts, serves as our ground truth and will allow us to validate our automated segmentation.

Evaluation is done using the methods provided for the practical work, refer to part 1 above for more detail.

\section{Method}
The method is strongly inspired by the article "Blood Vessels Segmentation in Retinal Fundus Image using Hybrid Method of Frangi Filter, Otsu Thresholding and Morphology" (accessible here: https://thesai.org/Downloads/Volume10No6/Paper_54-Blood_Vessels_Segmentation.pdf)
The method involves passing each image through three stages: pre-processing, segmentation, and morphological processing. Each stage consists of several operations, described below.
The method also includes a testing phase that we have also implemented to serve as a point of comparison to the validation method provided with the practical work.

However, we have modified the method to achieve what we believe are better results. For this, we replaced the last step with another pruning step. The comparison between the results obtained is made in the results section.

\section{Results}
\subsection{Precision and Recall}
\subsection{Skeletonization}
\subsection{Other Metrics}

\section{Conclusion}

\section{Bibliography}

\appendix
\section{Explored Path Without Success}

\end{document}
