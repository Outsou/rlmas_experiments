'''
.. py:module:: growing_tree
    :platform: Unix

Growing tree maze generation algorithm
'''
import random

import numpy


_nx = [0, 2, 0, -2]
_ny = [-2, 0, 2, 0]
_card = ['N', 'E', 'S', 'W']

_wc = {'N': (0, -1), 'E': (1, 0), 'S': (0, 1), 'W': (-1, 0)}
_opp = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

_first_probability = 0.9

def _create_maze_template(width, height):
    '''Create a maze template with **(width, height)** rooms.
    '''
    return numpy.zeros((width*2 + 1, height*2 + 1))


def room2xy(room):
    '''Return the pixel coordinate in the maze template for the given maze room
    coordinate.

    Pixel coordinate is computed as (2x + 1, 2y + 1) where (x, y) is the room
    coordinate
    '''
    return (room[0]*2 + 1, room[1]*2 + 1)


def random_room(maze):
    '''Return a random room for a given maze template.

    To reference the actual pixel value in the maze template, use
    :func:`room2xy`.
    '''
    w = (maze.shape[0] - 1) / 2
    h = (maze.shape[1] - 1) / 2
    return (random.randint(0, w - 1), random.randint(0, h - 1))


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


def choose_first(C):
    '''Choose the first item from the list.
    '''
    return C[0]


def choose_last(C):
    '''Choose the last item from the list.
    '''
    return C[-1]


def choose_random(C):
    '''Choose a random item from the list.
    '''
    return random.choice(C)

def choose_with_probability(C):
    rand = numpy.random.rand()

    if rand < _first_probability:
        return choose_first(C)
    else:
        return choose_last(C)


def is_visited(maze_tmpl, xy):
    if maze_tmpl[xy] == 0:
        return False
    return True


def create(x, y, choose_cell):
    '''Create a maze template with **(x, y)** rooms.

    The returned maze is a **(2x + 1, 2y + 1)** pixel black and white image,
    where each room is a white pixel at uneven coordinate ((1,1), (1,3), ...,
    etc.). The corridors between the rooms are also white. Impassable areas are
    black.

    :param int x: Width of the maze in rooms
    :param int y: Height of the maze in rooms
    :param callable choose_cell:
        Cell choosing method. Should be callable accepting one argument, which
        is the list of current cells to choose from. The callable should
        return one cell from the list. See, e.g. :func:`choose_last`, 
        :func:`choose_first`, :func:`choose_random`.

    :returns: Created maze
    '''
    maze = _create_maze_template(x, y)
    C = [room2xy(random_room(maze))]
    while len(C) > 0:
        xy = choose_cell(C)
        nbs = get_neighbors(xy, maze.shape[0], maze.shape[1])
        found = False
        for card, nb in nbs:
            if is_visited(maze, nb):
                pass
            else:
                found = True
                # Color the wall pixel between the rooms also.
                cxy = (xy[0] + _wc[card][0], xy[1] + _wc[card][1])
                maze[cxy] = 1.0
                maze[nb] = 1.0
                C.append(nb)
                break
        if not found:
            C.remove(xy)
    return maze


if __name__ == "__main__":
    x = 40
    y = 40
    import time
    t = time.time()
    N_MAZES = 1
    for _ in range(N_MAZES):
        maze = create(x, y, choose_with_probability)
    td = time.time() - t
    print("Total: {} ({} / maze)".format(td, td / N_MAZES))
    from matplotlib import pyplot as plt
    plt.imshow(maze, cmap='gray', interpolation=None)
    plt.show()


