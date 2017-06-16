from creamas.vote import VoteAgent
import mazes.growing_tree as gt
from artifacts.maze_artifact import MazeArtifact
import mazes.maze_solver as ms
from rl.bandit_learner import BanditLearner

import logging
import aiomas
import editdistance


class MazeAgent(VoteAgent):
    def __init__(self, environment, choose_func, maze_shape = (40, 40), search_width=10, rand=False, critic_threshold = 10, veto_threshold = 10, log_folder=None, log_level=logging.INFO, memsize=100):
        super().__init__(environment, log_folder=log_folder,
                         log_level=log_level)

        self.stmem = STMemory(length=memsize, max_length=memsize)
        self.maze_shape = maze_shape
        self.age = 0
        self._own_threshold = critic_threshold
        self._novelty_threshold = veto_threshold
        self.search_width = search_width
        self.bandit_learner = None
        self.rand = False
        self.connection_counts = None
        self.choose_func = choose_func

    def create(self, size_x, size_y, choose_cell):
        maze = gt.create(size_x, size_y, choose_cell)
        start = gt.room2xy(gt.random_room(maze))
        goal = gt.room2xy(gt.random_room(maze))
        node, expanded, added = ms.solver(maze, start, goal)
        path = ms.get_path(node)
        maze = ms.draw_path(maze, path)
        solution = ms.path_to_string(path)
        return {'maze': maze,
                'start': start,
                'goal': goal,
                'solution': solution}

    def novelty(self, maze):
        distance = self.stmem.distance(maze)
        return distance

    def evaluate(self, artifact):
        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        return self.novelty(artifact.obj), None

    def invent(self, n):
        maze = self.create(self.maze_shape[0], self.maze_shape[1], self.choose_func)
        best_artifact = MazeArtifact(self, maze, domain='maze')
        ev, _ = self.evaluate(best_artifact)
        best_artifact.add_eval(self, ev)
        for i in range(n-1):
            maze = self.create(self.maze_shape[0], self.maze_shape[1], self.choose_func)
            artifact = MazeArtifact(self, maze, domain='maze')
            ev, _ = self.evaluate(artifact)
            artifact.add_eval(self, ev)
            if ev > best_artifact.evals[self.name]:
                best_artifact = artifact

        best_artifact.creation_time = self.age
        best_artifact.self_criticism = 'reject'

        return best_artifact

    def learn(self, artifact, iterations=1):
        '''Train short term memory with given spirograph.
        :param spiro:
            :py:class:`SpiroArtifact` object
        '''
        for i in range(iterations):
            self.stmem.train_cycle(artifact.obj)

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)
        length = len(self.connections)
        self.bandit_learner = BanditLearner(length)

        self.connection_counts = {}
        for conn in conns:
            self.connection_counts[conn[0]] = 0

        return rets

    @aiomas.expose
    async def get_criticism(self, artifact):
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._novelty_threshold:
            artifact.add_eval(self, evaluation)
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    def get_connection_counts(self):
        return self.connection_counts

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    async def act(self):
        self.age += 1
        self.bandit_learner.increment_iteration_count()

        artifact = self.invent(self.search_width)

        val = artifact.evals[self.name]
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.learn(artifact, 1)

            bandit = self.bandit_learner.choose_bandit(rand=self.rand)
            critic = self.connections[bandit]
            self.connection_counts[critic] += 1

            connection = await self.env.connect(critic)
            passed, artifact = await connection.get_criticism(artifact)

            if passed:
                self.bandit_learner.give_reward(bandit, -1)
                self.env.add_candidate(artifact)
                self.added_last = True
            else:
                self.bandit_learner.give_reward(bandit, 1)


class STMemory():

    '''Agent's short-term memory model using a simple list which stores
    artifacts as is.'''
    def __init__(self, length, max_length = 100):
        self.length = length
        self.artifacts = []
        self.max_length = max_length

    def _add_artifact(self, artifact):
        if len(self.artifacts) >= 2 * self.max_length:
            self.artifacts = self.artifacts[:self.max_length]
        self.artifacts.insert(0, artifact)

    def learn(self, artifact):
        '''Learn new artifact. Removes last artifact from the memory if it is
        full.'''
        self._add_artifact(artifact)

    def train_cycle(self, artifact):
        '''Train cycle method to keep the interfaces the same with the SOM
        implementation of the short term memory.
        '''
        self.learn(artifact)

    def distance(self, artifact):

        if len(self.artifacts) == 0:
            return len(artifact['solution'])

        limit = self.get_comparison_amount()

        mdist = editdistance.eval(artifact['solution'], self.artifacts[0]['solution'])

        for i in range(1, limit):
            dist = editdistance.eval(artifact['solution'], self.artifacts[i]['solution'])
            if dist < mdist:
                mdist = dist
        return mdist

    def get_comparison_amount(self):
        if len(self.artifacts) < self.length:
            amount = len(self.artifacts)
        else:
            amount = self.length
        return amount
