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

    maze1 = create(size, size, choose_first)
    maze2 = create(size, size, choose_last)
    maze3 = create(size, size, choose_random)

    for i in range(amount):
        distance = np.sqrt(np.sum(np.square(maze1 - mazes1[i])))
        print(distance)

    print()

    for i in range(amount):
        distance = np.sqrt(np.sum(np.square(maze2 - mazes2[i])))
        print(distance)

    print()

    for i in range(amount):
        distance = np.sqrt(np.sum(np.square(maze3 - mazes3[i])))
        print(distance)


    plt.imshow(maze1, cmap='gray', interpolation=None)
    plt.figure()
    plt.imshow(maze2, cmap='gray', interpolation=None)
    plt.figure()
    plt.imshow(maze3, cmap='gray', interpolation=None)
    plt.show()

