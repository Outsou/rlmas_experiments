from creamas.vote import VoteAgent
from creamas.math import gaus_pdf
import mazes.growing_tree as gt
from artifacts.maze_artifact import MazeArtifact
import mazes.maze_solver as ms
from rl.bandit_learner import BanditLearner

import logging
import aiomas
import editdistance
import time


class MazeAgent(VoteAgent):
    def __init__(self, environment, choose_func = None, desired_novelty=-1, ask_criticism=True, maze_shape=(40, 40), search_width=10, ask_random=False, critic_threshold=10, veto_threshold=10, log_folder=None, log_level=logging.INFO, memsize=100):
        super().__init__(environment, log_folder=log_folder,
                         log_level=log_level)

        self.stmem = STMemory(length=memsize, max_length=memsize)
        self.maze_shape = maze_shape
        self.age = 0
        self._own_threshold = critic_threshold
        self._novelty_threshold = veto_threshold
        self.search_width = search_width
        self.bandit_learner = None
        self.ask_random = ask_random
        self.connection_counts = None
        self.choose_func = choose_func
        self.ask_criticism = ask_criticism
        self.comparison_count = 0
        self.artifacts_created = 0
        self.gatekeepers = None
        self.desired_novelty = desired_novelty

    def create(self, size_x, size_y, choose_cell):
        self.artifacts_created += 1
        maze = gt.create(size_x, size_y, choose_cell)
        start = gt.room2xy(gt.random_room(maze))
        goal = gt.room2xy(gt.random_room(maze))
        node, expanded, added = ms.solver(maze, start, goal)
        path = ms.get_path(node)
        #maze = ms.draw_path(maze, path)
        solution = ms.path_to_string(path)
        obj = {'maze': maze,
                'start': start,
                'goal': goal,
                'path': path,
                'function': choose_cell,
                'solution': solution}
        return obj

    def complexity(self, artifact):
        turns = ms.count_turns(artifact['maze'])
        turns = turns / self._get_room_count(artifact) # normalize
        return turns

    def novelty(self, artifact):
        self.comparison_count += self.stmem.get_comparison_amount()
        distance = self.stmem.distance(artifact)
        distance = distance / self._get_room_count(artifact) # normalize
        return distance

    @staticmethod
    def hedonic_value(value, desired_value):
        lmax = gaus_pdf(desired_value, desired_value, 4)
        pdf = gaus_pdf(value, desired_value, 4)
        return pdf / lmax

    def _get_room_count(self, artifact):
        w = int((artifact['maze'].shape[0] - 1) / 2)
        h = int((artifact['maze'].shape[1] - 1) / 2)
        return w*h

    def evaluate(self, artifact):
        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        novelty = self.novelty(artifact.obj)

        if self.desired_novelty > 0:
            evaluation = self.hedonic_value(novelty, self.desired_novelty)
        else:
            evaluation = novelty

        artifact.add_eval(self, evaluation)

        return evaluation, None

    def invent(self, n):
        maze = self.create(self.maze_shape[0], self.maze_shape[1], self.choose_func)
        best_artifact = MazeArtifact(self, maze, domain='maze')
        ev, _ = self.evaluate(best_artifact)
        for i in range(n-1):
            maze = self.create(self.maze_shape[0], self.maze_shape[1], self.choose_func)
            artifact = MazeArtifact(self, maze, domain='maze')
            ev, _ = self.evaluate(artifact)
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
        self.gatekeepers = list(self._connections.keys())
        length = len(self.gatekeepers)
        self.bandit_learner = BanditLearner(length)

        self.connection_counts = {}
        for conn in conns:
            self.connection_counts[conn[0]] = 0

        return rets

    @aiomas.expose
    async def get_criticism(self, artifact):
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._novelty_threshold:
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    def get_connection_counts(self):
        return self.connection_counts

    @aiomas.expose
    def get_comparison_count(self):
        return self.comparison_count

    @aiomas.expose
    def get_artifacts_created(self):
        return self.artifacts_created

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

            if not self.ask_criticism:
                self.add_candidate(artifact)
                self.added_last = True
                return

            bandit = self.bandit_learner.choose_bandit(rand=self.ask_random)
            critic = self.gatekeepers[bandit]
            self.connection_counts[critic] += 1

            connection = await self.env.connect(critic)
            passed, artifact = await connection.get_criticism(artifact)

            if passed:
                self.bandit_learner.give_reward(bandit, -1)
                self.add_candidate(artifact)
                self.added_last = True
            else:
                self.bandit_learner.give_reward(bandit, 1)

    @aiomas.expose
    def validate(self, candidates):
        valid_candidates = []

        for candidate in candidates:
            if self.name in candidate.evals:
                evaluation = candidate.evals[self.name]
            else:
                evaluation, _ = self.evaluate(candidate)

            if evaluation >= self._novelty_threshold:
                valid_candidates.append(candidate)

        return valid_candidates


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
