import pickle
import statistics
import numpy as np
from utilities.math import gini

def analyze(file):
    stats = pickle.load(open(file, "rb"))

    print('Means:')

    for key, value in stats.items():
        print('{}: {}'.format(key, statistics.mean(value)))

    print('\nGini coefficients:')

    for key, value in stats.items():
        print('{}: {}'.format(key, gini(np.array(value).astype(float))))




