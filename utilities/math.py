import numpy as np

def softmax(values, temperature):
    e_x = np.exp((values - np.max(values)) / temperature)
    return e_x / e_x.sum()