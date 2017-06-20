import pickle
import statistics
import numpy as np
from utilities.math import gini

def analyze(file):
    stats = pickle.load(open(file, "rb"))

    juttu = next (iter (stats.values()))
    print('Number of simulations: {}\n'.format(len(juttu)))

    print('Means:')

    items = sorted(stats.items())

    for key, value in items:
        if all(isinstance(x, (float, int)) for x in value):
            print('{}: {}'.format(key, statistics.mean(value)))

    # print('\nGini coefficients:')
    #
    # for key, value in items:
    #     if all(isinstance(x, (float, int)) for x in value):
    #         print('{}: {}'.format(key, gini(np.array(value).astype(float))))




