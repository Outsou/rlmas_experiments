import mazes.growing_tree as gt
import mazes.maze_solver as ms
from utilities.math import levenshtein

from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    maze1_method = gt.choose_first
    maze2_method = gt.choose_first

    maze1 = gt.create(40, 40, maze1_method)
    start = (1, 1)
    goal = (1, 3)
    node, expanded, added = ms.solver(maze1, start, goal)
    path = ms.get_path(node)
    maze1 = ms.draw_path(maze1, path)
    maze1_solution = ms.path_to_string(path)
    print(maze1_solution)

    fig = plt.figure()
    fig.add_subplot(2, 5, 1)
    plt.title('Len: ' + str(len(maze1_solution)))
    plt.imshow(maze1, cmap='gray', interpolation=None)

    for i in range(9):
        maze2 = gt.create(40, 40, maze2_method)
        start = gt.room2xy(gt.random_room(maze2))
        goal = gt.room2xy(gt.random_room(maze2))
        node, expanded, added = ms.solver(maze2, start, goal)
        path = ms.get_path(node)
        maze2 = ms.draw_path(maze2, path)

        maze2_solution = ms.path_to_string(path)
        print(maze2_solution)
        fig.add_subplot(2, 5, i+2)
        plt.title('Len dif: {}, Distance: {}'.format(abs(len(maze1_solution) - len(maze2_solution)), levenshtein(maze1_solution, maze2_solution)))
        plt.imshow(maze2, cmap='gray', interpolation=None)

    plt.show()







