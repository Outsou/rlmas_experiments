'''
.. py:module:: maze_solver
    :platform: Unix

A* maze solver for black and white images where white pixels are passable and
blacks are not.
'''
import itertools
import heapq
import math
import time
import random
import numpy as np
import matplotlib
import matplotlib.patches as patches

import mazes.growing_tree as gt

pq = []                         # list of entries arranged in a heap
entry_finder = {}               # mapping of tasks to entries
REMOVED = '<removed-task>'      # placeholder for a removed task
counter = itertools.count()     # unique sequence count

_wx = [0, 1, 0, -1]
_wy = [-1, 0, 1, 0]
_nx = [0, 2, 0, -2]
_ny = [-2, 0, 2, 0]


class AStarNode():

    def __init__(self, xy, last_node, h, cost):
        self.xy = xy
        self.last = last_node
        self.h = h
        self.cost = cost
        self.priority = h + cost
        self.visited = False

    def __lt__(self, anode):
        return self.priority < anode.priority


class PQ():

    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.counter = itertools.count()
        self.REMOVED = '<removed-task>'

    def add_task(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task.xy in self.entry_finder:
            self.remove_task(task.xy)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task.xy] = entry
        heapq.heappush(self.pq, entry)

    def remove_task(self, xy):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(xy)
        entry[-1] = self.REMOVED

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heapq.heappop(self.pq)
            if task is not self.REMOVED:
                del self.entry_finder[task.xy]
                return task
        raise KeyError('pop from an empty priority queue')


def h(xy, goal):
    return math.sqrt((xy[0]-goal[0])**2 + (xy[1] - goal[1])**2)


def get_nbs(maze, xy):
    nbs = []
    for i in range(4):
        cxy = (xy[0] + _wx[i], xy[1] + _wy[i])
        nxy = (xy[0] + _nx[i], xy[1] + _ny[i])
        if maze[cxy] == 1.0 and maze[nxy] == 1.0:
            nbs.append(nxy)
    random.shuffle(nbs)
    return nbs


def check_pq(pq, best_cost, goal):
    best_node = None
    while pq:
        try: 
            anode = pq.pop_task()
            if anode.xy == goal:
                if anode.cost < best_cost:
                    best_cost = anode.cost
                    best_node = anode
        except KeyError:
            return best_node
    return best_node


def solver(maze, start, goal):
    '''A maze solver using A*-search algorithm.

    :param maze: Numpy 2D binary array.
    :param start: coordinate of the starting room.
    :param goal: coordinate of the goal room.

    :returns:
        :class:`AStarNode` which is the node in the goal room. The whole path
        from *start* to *goal* can be constructed by iterating over the
        :attr:`last` of the node until it is *None*.
    '''
    pq = PQ()
    visited = {}
    node = AStarNode(start, None, h(start, goal), 0.0)
    pq.add_task(node, node.priority)
    found = False
    nodes_added = 1
    nodes_expanded = 0
    while pq and not found:
        anode = pq.pop_task()
        anode.visited = True
        nodes_expanded += 1
        visited[anode.xy] = anode
        if anode.xy == goal:
            found = True
            bnode = check_pq(pq, anode.cost, goal)
            if bnode is not None:
                return bnode, nodes_expanded, nodes_added
            else:
                return anode, nodes_expanded, nodes_added
        nbs = get_nbs(maze, anode.xy)
        for xy in nbs:
            if xy in visited:
                continue
            nnode = AStarNode(xy, anode, h(xy, goal), anode.cost + 2)
            #print("Handling: {} {} {}".format(xy, nnode.h, nnode.cost))
            if xy in pq.entry_finder:
                _, _, cnode = pq.entry_finder[xy]
                if nnode.priority < cnode.priority:
                    pq.add_task(nnode, nnode.priority)
                    nodes_added += 1
            else:
                pq.add_task(nnode, nnode.priority)
                nodes_added += 1


def draw_path(maze, path, color=0.7):
    for i, xy in enumerate(path[:-1]):
        maze[xy] = color
        wx = int((xy[0] + path[i + 1][0]) / 2)
        wy = int((xy[1] + path[i + 1][1]) / 2)
        maze[(wx, wy)] = color
    maze[path[-1]] = color
    return maze

def calculate_decisions(maze, path):
    decisions = 0
    came_from = None
    for i, xy in enumerate(path[:-1]):
        x = xy[0]
        y = xy[1]
        coords = (x-1, y)
        if maze[coords] > 0 and coords != came_from:
            decisions += 1
        coords = (x+1, y)
        if maze[coords] > 0 and coords != came_from:
            decisions += 1
        coords = (x, y-1)
        if maze[coords] > 0 and coords != came_from:
            decisions += 1
        coords = (x, y+1)
        if maze[coords] > 0 and coords != came_from:
            decisions += 1
        decisions -= 1

        wx = int((xy[0] + path[i + 1][0]) / 2)
        wy = int((xy[1] + path[i + 1][1]) / 2)
        came_from = (wx, wy)
    return decisions

def calculate_turns_in_path(maze, path_string):
    turns = 0
    prev_char = path_string[0]
    for char in path_string[1:]:
        if char != prev_char:
            turns += 1
        prev_char = char
    return turns

def get_path(node):
    path = []
    while node is not None:
        path.append(node.xy)
        node = node.last
    path.reverse()
    return path

def path_to_string(path):
    s = ''

    for i in range(len(path)-1):
        movement = np.subtract(path[i], path[i+1])
        if movement[0] < 0:
            s += 'S'
        elif movement[0] > 0:
            s += 'N'
        elif movement[1] < 0:
            s += 'E'
        else:
            s += 'W'

    return s

def count_turns(maze):
    w = int((maze.shape[0] - 1) / 2)
    h = int((maze.shape[1] - 1) / 2)
    count = 0

    for x in range(w):
        for y in range(h):
            # count open paths
            open_paths = 0
            room_xy = gt.room2xy((x, y))
            n_open = False
            s_open = False
            e_open = False
            w_open = False
            if maze[room_xy[0] - 1, room_xy[1]] > 0:
                n_open = True
                open_paths += 1
            if maze[room_xy[0] + 1, room_xy[1]] > 0:
                s_open = True
                open_paths += 1
            if maze[room_xy[0], room_xy[1] - 1] > 0:
                w_open = True
                open_paths += 1
            if maze[room_xy[0], room_xy[1] + 1] > 0:
                e_open = True
                open_paths += 1

            if open_paths == 2:
                if not ((n_open and s_open) or (e_open and w_open)):
                    count += 1

    return count


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    t = time.time()
    N_MAZES = 100
    tmin = 10000
    tmax = 0
    solved_sum = 0.0
    created_sum = 0.0
    for i in range(N_MAZES):
        # #print("Maze {}".format(i+1))
        # t2 = time.time()
        # maze = gt.create(40, 40, gt.choose_last)
        # start = gt.room2xy(gt.random_room(maze))
        # goal = gt.room2xy(gt.random_room(maze))
        # #start = (1,1)
        # #goal = (79, 79)
        # t3 = time.time()
        # t_created = t3 - t2
        # node, expanded, added = solver(maze, start, goal)
        # t_solved = time.time() - t3
        # t_total = time.time() - t2
        # created_sum += t_created
        # solved_sum += t_solved
        # print("Maze {:0>4} created in {:.4f}, solved {}=>{} in {:.4f}. Total {:.4f}"
        #       .format(i+1, t_created, start, goal, t_solved, t_total))
        # path = get_path(node)
        # print("Path length {}, nodes expanded {}, nodes added {}."
        #       .format(len(path), expanded, added))
        # print("Number of decisions: {}"
        #       .format(calculate_decisions(maze, path)))
        #
        # path_string = path_to_string(path)
        # print("Number of turns: {}"
        #       .format(calculate_turns_in_path(maze, path_string)))



        import pickle
        import zlib
        import sys

        gt._first_probability = 0.9
        for cell_chooser in [gt.choose_first, gt.choose_with_probability, gt.choose_last]:
            maze = gt.create(40, 40, cell_chooser)
            print('Turns: {}'.format(count_turns(maze)))
            plt.imshow(maze, cmap='gray', interpolation=None)
            plt.show()

        for prob in reversed([0.1, 0.3, 0.5, 0.7, 0.9]):
            gt._first_probability = prob
            maze = gt.create(40, 40, gt.choose_with_probability)
            print('Probability: {}'.format(prob))
            print('Turns: {}'.format(count_turns(maze)))
            plt.imshow(maze, cmap='gray', interpolation=None)
            plt.show()

        # maze = draw_path(maze, path)
        #
        # value = 0.7
        #
        # masked_array = np.ma.masked_where(maze == value, maze)
        #
        # cmap = matplotlib.cm.gray
        # cmap.set_bad(color='blue')
        #
        # fig, ax = plt.subplots(1)
        # ax.imshow(masked_array, cmap=cmap, interpolation=None)
        # start_circle = patches.Circle((start[1], start[0]), 2, linewidth=2, edgecolor='y', facecolor='none')
        # ax.add_patch(start_circle)
        # goal_circle = patches.Circle((goal[1], goal[0]), 2, linewidth=2, edgecolor='r', facecolor='none')
        # ax.add_patch(goal_circle)
        # plt.show()


    td = time.time() - t
    print("Overall {} mazes generated and solved in: {} ({} / maze)"
          .format(N_MAZES, td, td / N_MAZES))
    print("Total generation time {}. ({} / maze)"
          .format(created_sum, created_sum / N_MAZES))
    print("Total solving time {}. ({} / maze)"
          .format(solved_sum, solved_sum / N_MAZES))

