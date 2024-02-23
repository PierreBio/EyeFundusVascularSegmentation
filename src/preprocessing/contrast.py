from PIL import Image
import numpy as np

def enhance_contrast(img, level = 256, blocks = 8, threshold = 10.0):
    """
    Enhances the contrast of an image using CLAHE.

    Args:
        img (Image.Image): The input image.
        level (int): Number of intensity levels (default 256).
        blocks (int): The image is divided into blocks*blocks sections for localized contrast enhancement.
        threshold (float): Clipping threshold for histogram equalization (default 10.0).

    Returns:
        Image.Image: The contrast-enhanced image.
    """
    img_arr = np.array(img)
    if len(img_arr.shape) == 2:
        channel_num = 1
    elif len(img_arr.shape) == 3:
        channel_num = img_arr.shape[2]

    if channel_num == 1:
        arr = contrast_limited_ahe(img_arr, level = level, blocks = blocks, threshold = threshold)
        img_res = Image.fromarray(arr)
    elif channel_num == 3 or channel_num == 4:
        # RGB image or RGBA image(such as png)
        rgb_arr = [None] * 3
        rgb_img = [None] * 3
        # process dividely
        for k in range(3):
            rgb_arr[k] = contrast_limited_ahe(img_arr[:,:,k], level = level, blocks = blocks, threshold = threshold)
            rgb_img[k] = Image.fromarray(rgb_arr[k])
        img_res = Image.merge("RGB", tuple(rgb_img))

    return img_res

def contrast_limited_ahe(img_arr, level = 256, blocks = 8, threshold = 10.0, **args):
    """
    equalize the distribution of histogram to enhance contrast, using CLAHE.

    Args:
        img_array (numpy.ndarray): 2D image array.
        level (int): Number of intensity levels.
        blocks (int): The image is divided into blocks*blocks sections.
        threshold (float): Clipping threshold for histogram equalization.

    Returns:
        numpy.ndarray: CLAHE processed image array.
    """
    (m, n) = img_arr.shape
    block_m = int(m / blocks)
    block_n = int(n / blocks)

    # split small regions and calculate the CDF for each, save to a 2-dim list
    maps = []
    for i in range(blocks):
        row_maps = []
        for j in range(blocks):
            # block border
            si, ei = i * block_m, (i + 1) * block_m
            sj, ej = j * block_n, (j + 1) * block_n

            # block image array
            block_img_arr = img_arr[si : ei, sj : ej]

            # calculate histogram and cdf
            hists = calc_histogram_(block_img_arr)
            clip_hists = clip_histogram_(hists, threshold = threshold)     # clip histogram
            hists_cdf = calc_histogram_cdf_(clip_hists, block_m, block_n, level)

            # save
            row_maps.append(hists_cdf)
        maps.append(row_maps)

    # interpolate every pixel using four nearest mapping functions
    # pay attention to border case
    arr = img_arr.copy()
    for i in range(m):
        for j in range(n):
            r = int((i - block_m / 2) / block_m)      # the row index of the left-up mapping function
            c = int((j - block_n / 2) / block_n)      # the col index of the left-up mapping function

            x1 = (i - (r + 0.5) * block_m) / block_m  # the x-axis distance to the left-up mapping center
            y1 = (j - (c + 0.5) * block_n) / block_n  # the y-axis distance to the left-up mapping center

            lu = 0    # mapping value of the left up cdf
            lb = 0    # left bottom
            ru = 0    # right up
            rb = 0    # right bottom

            # four corners use the nearest mapping directly
            if r < 0 and c < 0:
                arr[i][j] = maps[r + 1][c + 1][img_arr[i][j]]
            elif r < 0 and c >= blocks - 1:
                arr[i][j] = maps[r + 1][c][img_arr[i][j]]
            elif r >= blocks - 1 and c < 0:
                arr[i][j] = maps[r][c + 1][img_arr[i][j]]
            elif r >= blocks - 1 and c >= blocks - 1:
                arr[i][j] = maps[r][c][img_arr[i][j]]
            # four border case using the nearest two mapping : linear interpolate
            elif r < 0 or r >= blocks - 1:
                if r < 0:
                    r = 0
                elif r > blocks - 1:
                    r = blocks - 1
                left = maps[r][c][img_arr[i][j]]
                right = maps[r][c + 1][img_arr[i][j]]
                arr[i][j] = (1 - y1) * left + y1 * right
            elif c < 0 or c >= blocks - 1:
                if c < 0:
                    c = 0
                elif c > blocks - 1:
                    c = blocks - 1
                up = maps[r][c][img_arr[i][j]]
                bottom = maps[r + 1][c][img_arr[i][j]]
                arr[i][j] = (1 - x1) * up + x1 * bottom
            # bilinear interpolate for inner pixels
            else:
                lu = maps[r][c][img_arr[i][j]]
                lb = maps[r + 1][c][img_arr[i][j]]
                ru = maps[r][c + 1][img_arr[i][j]]
                rb = maps[r + 1][c + 1][img_arr[i][j]]
                arr[i][j] = (1 - y1) * ( (1 - x1) * lu + x1 * lb) + y1 * ( (1 - x1) * ru + x1 * rb)
    arr = arr.astype("uint8")
    return arr

def calc_histogram_(gray_arr, level = 256):
    """
    Calculates the histogram of a grayscale image array.

    Args:
        gray_array (numpy.ndarray): 2D grayscale image array.
        level (int): Number of intensity levels.

    Returns:
        list: Histogram of the image array.
    """
    hists = [0 for _ in range(level)]
    for row in gray_arr:
        for p in row:
            hists[p] += 1
    return hists

def calc_histogram_cdf_(hists, block_m, block_n, level = 256):
    """
    Calculates the CDF based on a histogram.

    Args:
        histogram (list): Histogram of a block of the image.
        block_height (int): Height of the block.
        block_width (int): Width of the block.
        level (int): Number of intensity levels.

    Returns:
        numpy.ndarray: CDF of the histogram.
    """
    hists_cumsum = np.cumsum(np.array(hists))
    const_a = (level - 1) / (block_m * block_n)
    hists_cdf = (const_a * hists_cumsum).astype("uint8")
    return hists_cdf

def clip_histogram_(hists, threshold = 10.0):
    """
    Clips a histogram based on a specified threshold.

    Args:
        histogram (list): Histogram to be clipped.
        threshold (float): Clipping threshold as a multiplier of the mean histogram value.

    Returns:
        list: The clipped histogram.
    """
    all_sum = sum(hists)
    threshold_value = all_sum / len(hists) * threshold
    total_extra = sum([h - threshold_value for h in hists if h >= threshold_value])
    mean_extra = total_extra / len(hists)

    clip_hists = [0 for _ in hists]
    for i in range(len(hists)):
        if hists[i] >= threshold_value:
            clip_hists[i] = int(threshold_value + mean_extra)
        else:
            clip_hists[i] = int(hists[i] + mean_extra)

    return clip_hists