import numpy as np
import matplotlib.pyplot as plt

def total_pix(image):
    size = image.shape[0] * image.shape[1]
    return size

def histogramify(image):
    grayscale_array = []
    for w in range(0,image.size[0]):
        for h in range(0,image.size[1]):
            intensity = image.getpixel((w,h))
            grayscale_array.append(intensity)

    total_pixels = image.size[0] * image.size[1]
    bins = range(0,257)
    img_histogram = np.histogram(grayscale_array, bins)
    return img_histogram


def otsu(image, threshold):
    hist = histogramify(image) # get hist
    total = total_pix(image) # get total size
    sumT, sum0, sum1 = 0, 0, 0
    w0, w1 = 0, 0
    varBetween, mean0, mean1 = 0, 0, 0
    for i in range(0,256):
        sumT += i * hist[0][i]
        if i < threshold:
            w0 += hist[0][i] # num of under threshold's pixel
            sum0 += i * hist[0][i]
    w1 = total - w0
    if w1 == 0:
        return 0

    sum1 = sumT - sum0
    mean0 = sum0/(w0*1.0)
    mean1 = sum1/(w1*1.0)
    varBetween = w0/(total*1.0) * w1/(total*1.0) * (mean0-mean1)*(mean0-mean1) # formulation form https://en.wikipedia.org/wiki/Otsu%27s_method
    # print "varBetween is:", varBetween
    return varBetween


def fast_ostu(image, threshold):
    image = np.transpose(np.asarray(image))
    total = total_pix(image)
    bin_image = image<threshold
    sumT = np.sum(image)
    w0 = np.sum(bin_image)
    sum0 = np.sum(bin_image * image)
    w1 = total - w0
    if w1 == 0:
        return 0
    sum1 = sumT - sum0
    mean0 = sum0 / (w0 * 1.0)
    mean1 = sum1 / (w1 * 1.0)
    varBetween = w0 / (total * 1.0) * w1 / (total * 1.0) * (mean0 - mean1) * (
            mean0 - mean1)  # formulation form https://en.wikipedia.org/wiki/Otsu%27s_method
    # print "varBetween is:", varBetween
    return varBetween

def fast_otsu(image):
    # Convert image to a 1D array of pixel intensities
    pixels = image.ravel()

    # Calculate histogram and probabilities of each intensity level
    histogram, bin_edges = np.histogram(pixels, bins=256, range=(0, 256))
    pixel_probs = histogram / pixels.size

    # Cumulative sum and cumulative mean
    cumulative_sum = np.cumsum(histogram)
    cumulative_mean = np.cumsum(histogram * bin_edges[:-1])

    # Global intensity mean
    global_mean = cumulative_mean[-1]

    # Between-class variance for each threshold
    between_class_variance = ((global_mean * cumulative_sum - cumulative_mean)**2) / (cumulative_sum * (1 - cumulative_sum))

    # Add a print statement or plot here to check the variance
    print("Between class variance:", between_class_variance)
    plt.plot(between_class_variance)
    plt.title("Between Class Variance over Thresholds")
    plt.xlabel("Threshold")
    plt.ylabel("Variance")
    plt.show()

    # Find the threshold that maximizes the between-class variance
    optimal_threshold = np.argmax(between_class_variance)

    return optimal_threshold