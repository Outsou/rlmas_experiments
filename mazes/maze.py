'''
.. py:module:: maze
    :platform: Unix

Simple maze-generation example
'''
import random
import sys

sys.setrecursionlimit(10000)

_opp = {'N': 'S', 'E': 'W', 'S': 'N', 'W': 'E'}

class MazeNode():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.nb = {'N': None, 'E': None, 'S': None, 'W': None}
        self.walls = {'E': True, 'S': True}
        self.visited = False

    def __str__(self):
        s = ""
        if self.walls['S']:
            s += "_"
        else:
            s += " "
        if self.walls['E']:
            s += "|"
        else:
            if self.walls['S']:
                s += "_"
            else:
                s += " "
        return s

    def __repr__(self):
        return self.__str__()

def _populate_nbs(maze):
    for x in range(len(maze)):
        for y in range(len(maze[0])):
            node = maze[x][y]
            if x > 0:
                node.nb['W'] = maze[x-1][y]
            if x < len(maze) - 1:
                node.nb['E'] = maze[x+1][y]
            if y > 0:
                node.nb['N'] = maze[x][y-1]
            if y < len(maze[0]) - 1:
                node.nb['S'] = maze[x][y+1]


def _remove_walls(node, last_node_dir=None):
    if node.visited:
        return
    node.visited = True
    if last_node_dir == 'E':
        node.walls['E'] = False
    if last_node_dir == 'S':
        node.walls['S'] = False
    if last_node_dir == 'N':
        node.nb['N'].walls['S'] = False
    if last_node_dir == 'W':
        node.nb['W'].walls['E'] = False
    node_nb = list(node.nb.items())
    random.shuffle(node_nb)
    for d, n in node_nb:
        if n is not None:
            _remove_walls(n, _opp[d])


def get_str(maze):
    s = "_"
    for x in range(len(maze)):
        s += "__"
    s += "\n"
    for y in range(len(maze[0])):
        line = "|"
        for x in range(len(maze)):
            line += str(maze[x][y])
        if line[-1] == " ":
            line = line[:-1] + "|"
        s += line + "\n"
    return s


def create(x, y):
    '''Create a maze with given dimensions.
    '''
    assert x > 0
    assert y > 0
    maze = [[MazeNode(i, j) for j in range(y)] for i in range(x)]
    _populate_nbs(maze)
    rx = random.randint(0, x-1)
    ry = random.randint(0, y-1)
    rnode = maze[rx][ry]
    _remove_walls(rnode)
    return maze

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
    #print(get_str(maze))
