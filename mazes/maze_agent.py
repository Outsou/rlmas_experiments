'''
.. py:module:: maze_agent
    :platform: Unix

An agent that creates and solves mazes.
'''
import logging
import random
import time

import aiomas
from creamas.grid import GridAgent, GridEnvironment 
from creamas import Artifact

import growing_tree as gt
import maze_solver as ms
from serializers import get_artifact_ser


class MazeAgent(GridAgent):
    '''A maze creating and solving agent.
    '''
    def __init__(self, maze_x=40, maze_y=40, choose_cell=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maze_x = maze_x
        self.maze_y = maze_y
        if choose_cell is None:
            self.choose_cell = gt.choose_last()
        else:
            self.choose_cell = choose_cell

    def create(self, x, y, choose_cell):
        '''Create a maze with the growing tree algorithm.
        '''
        maze = gt.create(x, y, choose_cell)
        start = gt.room2xy(gt.random_room(maze))
        goal = gt.room2xy(gt.random_room(maze))
        art = Artifact(self, (maze, start, goal))
        return art

    def solve(self, maze_artifact):
        '''Solve a maze with A*-search.
        '''
        maze, start, goal = maze_artifact.obj
        node, expanded, added = ms.solver(maze, start, goal)
        path = ms.get_path(node)
        self._log(logging.DEBUG, "Solved maze task {}->{}. Path length {}, "
                  "expanded {}, added {}".format(start, goal, len(path),
                                                 expanded, added))
        return path, expanded, added

    @aiomas.expose
    def evaluate(self, artifact):
        path, expanded, added = self.solve(artifact)
        return expanded / len(path)

    async def act(self):
        maze_art = self.create(self.maze_x, self.maze_y, self.choose_cell)
        maze, start, goal = maze_art.obj
        self._log(logging.DEBUG, "Created maze task {}->{}"
                  .format(start, goal))
        card = random.choice(['N', 'E', 'S', 'W'])
        addr = self.neighbors[card]
        if addr is not None:
            self._log(logging.DEBUG, "Asking opinion from {}.".format(addr))
            opinion = await self.ask_opinion(addr, maze_art)
            self._log(logging.DEBUG, "Got opinion {} from {} maze task {}->{}."
                      .format(opinion, addr, start, goal))


if __name__ == "__main__":
    env = GridEnvironment.create(('localhost', 5555), codec=aiomas.MsgPack,
                                 extra_serializers=[get_artifact_ser])
    env.gs = (10,10)
    env.origin = (0,0)

    for _ in range(100):
        agent = MazeAgent(40, 40, gt.choose_random, env, log_folder='logs',
                          log_level=logging.DEBUG)

    aiomas.run(env.set_agent_neighbors())

    STEPS = 5
    t = time.time()
    for i in range(STEPS):
        t1 = time.time()
        print("Step {}".format(i + 1))
        aiomas.run(env.trigger_all())
        t2 = time.time() - t1
        print("In {:.2f} seconds".format(t2))
    td = time.time() - t
    print("{} steps in {:.2f} seconds".format(STEPS, td))
    env.shutdown()