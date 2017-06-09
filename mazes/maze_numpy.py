'''
.. py:module:: maze_numpy
    :platform: Unix

Recursive maze generation using Numpy-array.
'''
import random
import sys

import numpy

sys.setrecursionlimit(10000)

_nx = [0, 2, 0, -2]
_ny = [-2, 0, 2, 0]
_card = ['N', 'E', 'S', 'W']

_wc = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
_opp = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

def _create_maze_template(width, height):
    return numpy.zeros((width*2 + 1, height*2 + 1))


def _room2xy(room):
    return (room[0]*2 + 1, room[1]*2 + 1)


def random_room(maze_tmpl):
    w = (maze_tmpl.shape[0] - 1) / 2
    h = (maze_tmpl.shape[1] - 1) / 2
    return (random.randint(0, w - 1), random.randint(0, h - 1))


def rand_neighbor(xy, w, h):
    acceptable = False
    while not acceptable:
        c = random.choice([0, 1, 2, 3])
        nx = xy[0] + _nx[c]
        ny = xy[1] + _ny[c]
        if nx > 0 and nx < w and ny > 0 and ny < h:
            acceptable = True
    return (nx, ny)


def get_neighbors(xy, w, h, randomize=True):
    nbs = []
    for i in range(4):
        nx = xy[0] + _nx[i]
        ny = xy[1] + _ny[i]
        if nx > 0 and nx < w and ny > 0 and ny < h:
            nbs.append((_card[i], (nx, ny)))
    if randomize:
        random.shuffle(nbs)
    return nbs


def is_visited(maze_tmpl, xy):
    if maze_tmpl[xy] == 0:
        return False
    return True

def _remove_walls(maze_tmpl, xy, last_dir=None):
    if is_visited(maze_tmpl, xy):
        return
    maze_tmpl[xy] = 1.0
    if last_dir is not None:
        maze_tmpl[(xy[0] + _wc[last_dir][0], xy[1] + _wc[last_dir][1])] = 1.0
    for card, nb in get_neighbors(xy, maze_tmpl.shape[0], maze_tmpl.shape[1]):
        _remove_walls(maze_tmpl, nb, last_dir=_opp[card])


def create(x, y):
    maze_tmpl = _create_maze_template(x, y)
    xy = _room2xy(random_room(maze_tmpl))
    _remove_walls(maze_tmpl, xy)
    return maze_tmpl

if __name__ == "__main__":
    x = 80
    y = 80
    import time
    t = time.time()
    N_MAZES = 100
    for _ in range(N_MAZES):
        maze = create(x, y)
    td = time.time() - t
    print("Total: {} ({} / maze)".format(td, td / N_MAZES))
    #from matplotlib import pyplot as plt
    #plt.imshow(maze, cmap='gray', interpolation=None)
    #plt.show()


