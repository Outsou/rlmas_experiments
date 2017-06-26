from agents.maze.maze_agent import MazeAgent
import mazes.maze_solver as ms
from artifacts.maze_artifact import MazeArtifact

import numpy as np
import aiomas
import logging

class CreatorAgent(MazeAgent):

    def __init__(self, environment, choose_funcs, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.name = self.name + '_cr_N' + str(self.desired_novelty)
        self.choose_funcs = choose_funcs
        self.choose_func_counts = {}
        for func in self.choose_funcs:
            self.choose_func_counts[func] = 0
        self.choose_func = np.random.choice(self.choose_funcs)
        self.published_count = 0

    @aiomas.expose
    def deliver_publication(self, artifact):
        evaluation, _ = self.evaluate(artifact)
        if artifact.creator == self.name:
            self.published_count += 1
        elif evaluation >= self._novelty_threshold:
            self.learn(artifact)
            self.choose_func = artifact.obj['function']

    @aiomas.expose
    def get_choose_func_counts(self):
        return self.choose_func_counts

    @aiomas.expose
    def get_memory_artifacts(self):
        return self.stmem.artifacts[:self.stmem.length]

    @aiomas.expose
    def get_published_count(self):
        return self.published_count

    @aiomas.expose
    async def act(self):
        self.age += 1
        self.bandit_learner.increment_iteration_count()

        artifact = self.invent(self.search_width)
        self.choose_func_counts[self.choose_func] += 1
        val = artifact.evals[self.name]
        self.add_artifact(artifact)

        novelty = self.novelty(artifact.obj)
        self._log(logging.INFO,
                  'Created artifact with novelty {}, hedonic value {} and solution length {} with function {}'
                  .format(novelty, val, len(artifact.obj['solution']), self.choose_func.__name__))

        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.passed_self_criticism_count += 1
            self.learn(artifact)

            if not self.ask_criticism:
                return

            bandit = self.bandit_learner.choose_bandit(rand=self.ask_random)
            gatekeeper = self.gatekeepers[bandit]
            self.connection_counts[gatekeeper] += 1
            connection = await self.env.connect(gatekeeper)
            passed, artifact = await connection.get_criticism(artifact, self.addr)

            if passed:
                self.bandit_learner.give_reward(bandit, 1)
            else:
                self.bandit_learner.give_reward(bandit, -1)
        else:
            choices = [func for func in self.choose_funcs if func != self.choose_func]
            if len(choices) > 0:
                self.choose_func = np.random.choice(choices)
