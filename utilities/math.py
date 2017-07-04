import numpy as np
import pylab as pl


def softmax(values, temperature):
    e_x = np.exp((values - np.max(values)) / temperature)
    return e_x / e_x.sum()


# Stolen from https://github.com/oliviaguest/gini
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1,array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def box_count(image):
    pixels = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                pixels.append((i, j))
    lx = image.shape[1]
    ly = image.shape[0]
    pixels = pl.array(pixels)
    if len(pixels) < 2:
        return 0
    scales = np.logspace(1, 4, num=20, endpoint=False, base=2)
    Ns = []
    for scale in scales:
        # print('Scale: ' + str(scale))
        H, edges = np.histogramdd(pixels, bins=(np.arange(0, lx, scale), np.arange(0, ly, scale)))
        H_sum = np.sum(H > 0)
        if H_sum == 0:
            H_sum = 1
        Ns.append(H_sum)

    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    hausdorff_dim = -coeffs[0]

    return hausdorff_dim
