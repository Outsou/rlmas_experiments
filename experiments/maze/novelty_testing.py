from mazes.growing_tree import create
from mazes.growing_tree import choose_last
from mazes.growing_tree import choose_first
from mazes.growing_tree import choose_random

from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':

    mazes1 = []
    mazes2 = []
    mazes3 = []

    size = 40
    amount = 10

    for _ in range(amount):
        mazes1.append(create(size, size, choose_first))
        mazes2.append(create(size, size, choose_last))
        mazes3.append(create(size, size, choose_random))

    test_mazes = []

    test_mazes.append(create(size, size, choose_first))
    test_mazes.append(create(size, size, choose_last))
    test_mazes.append(create(size, size, choose_random))

    for test_maze in test_mazes:
        for i in range(amount):
            distance = np.sqrt(np.sum(np.square(test_maze - mazes1[i])))
            print(distance)

        print()

        for i in range(amount):
            distance = np.sqrt(np.sum(np.square(test_maze - mazes2[i])))
            print(distance)

        print()

        for i in range(amount):
            distance = np.sqrt(np.sum(np.square(test_maze - mazes3[i])))
            print(distance)

        print()

        plt.imshow(test_maze, cmap='gray', interpolation=None)
        plt.figure()

    plt.show()

    # plt.imshow(maze1, cmap='gray', interpolation=None)
    # plt.figure()
    # plt.imshow(maze2, cmap='gray', interpolation=None)
    # plt.figure()
    # plt.imshow(maze3, cmap='gray', interpolation=None)
    # plt.show()

