from agents.maze.maze_agent import MazeAgent
import mazes.maze_solver as ms
from artifacts.maze_artifact import MazeArtifact

import numpy as np
import aiomas

class CreatorAgent(MazeAgent):

    def __init__(self, environment, choose_funcs, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.choose_funcs = choose_funcs
        self.choose_func_counts = {}
        for func in self.choose_funcs:
            self.choose_func_counts[func] = 0
        self.choose_func = np.random.choice(self.choose_funcs)

    @aiomas.expose
    def deliver_publication(self, artifact):
        evaluation = self.evaluate(artifact)
        if evaluation >= self._novelty_threshold:
            self.choose_func = artifact.obj['function']

    @aiomas.expose
    async def act(self):
        self.age += 1
        self.bandit_learner.increment_iteration_count()

        artifact = self.invent(self.search_width)
        self.choose_func_counts[self.choose_func] += 1

        val = artifact.evals[self.name]
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.learn(artifact, 1)

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
            self.choose_func = np.random.choice(choices)