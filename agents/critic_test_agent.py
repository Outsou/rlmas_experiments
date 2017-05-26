from creamas.examples.spiro.spiro_agent_mp import SpiroAgent
from rl.bandit_learner import BanditLearner

import aiomas
import numpy as np
import logging


class CriticTestAgent(SpiroAgent):

    def _init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)

    @aiomas.expose
    def set_acquaintances(self, addresses):
        addresses = list(addresses)
        addresses.remove(self.addr)

        self.acquaintances = []

        for acquaintance in addresses:
            self.acquaintances.append([acquaintance, 0])

        self.bandit_learner = BanditLearner(len(self.acquaintances))

    @aiomas.expose
    def ask_if_passes(self, artifact):
        evaluation, _ = self.evaluate(artifact)
        if evaluation >= self._novelty_threshold:
            return True
        else:
            return False

    @aiomas.expose
    async def act(self):
        artifact = self.invent(10)

        bandit = self.bandit_learner.choose_bandit()
        acquaintance = self.acquaintances[bandit]
        acquaintance[1] += 1

        connection = await self.env.connect(acquaintance[0])
        passed = await connection.ask_if_passes(artifact)

        if passed:
            self.bandit_learner.give_reward(bandit, -1)
        else:
            self.bandit_learner.give_reward(bandit, 1)

        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self.spiro_args = args
        self.arg_history.append(self.spiro_args)
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            # Train SOM with the invented artifact
            self.learn(artifact, self.teaching_iterations)
        elif self.jump == 'random':
            largs = self.spiro_args
            self.spiro_args = np.random.uniform(-199, 199,
                                                self.spiro_args.shape)
            self._log(logging.DEBUG, "Jumped from {} to {}"
                      .format(largs, self.spiro_args))

    def close(self, folder):
        pass
