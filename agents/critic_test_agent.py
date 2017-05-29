from creamas.examples.spiro.spiro_agent_mp import SpiroAgent
from rl.bandit_learner import BanditLearner

import aiomas
import numpy as np
import logging


class CriticTestAgent(SpiroAgent):

    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.comparison_count = 0
        self.name = "{}_M{}".format(self.name, self.stmem.length)

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

        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self.spiro_args = args
        self.arg_history.append(self.spiro_args)
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            # Train SOM with the invented artifact
            self.learn(artifact, self.teaching_iterations)
            # Ask someone for veto
            bandit = self.bandit_learner.choose_bandit()
            acquaintance = self.acquaintances[bandit]
            acquaintance[1] += 1

            connection = await self.env.connect(acquaintance[0])
            passed = await connection.ask_if_passes(artifact)

            if passed:
                self.bandit_learner.give_reward(bandit, -1)
            else:
                self.bandit_learner.give_reward(bandit, 1)
        elif self.jump == 'random':
            largs = self.spiro_args
            self.spiro_args = np.random.uniform(-199, 199,
                                                self.spiro_args.shape)
            self._log(logging.DEBUG, "Jumped from {} to {}"
                      .format(largs, self.spiro_args))

    @aiomas.expose
    def get_addr(self):
        return self.addr

    @aiomas.expose
    def get_acquaintances(self):
        return self.acquaintances

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    def get_comparison_count(self):
        return self.comparison_count

    @aiomas.expose
    def get_acquaintance_values(self):
        acquaintance_values = {}

        for i in range(len(self.acquaintances)):
            acquaintance_values[self.acquaintances[i][0]] = self.bandit_learner.bandits[i]

        return acquaintance_values

    def evaluate(self, artifact):
        '''Evaluate the artifact with respect to the agents short term memory.

        Returns value in [0, 1].
        '''

        # Keep track of comparisons
        self.comparison_count += len(self.stmem.artifacts)

        if self.desired_novelty > 0:
            return self.hedonic_value(self.novelty(artifact.obj))
        return self.novelty(artifact.obj) / self.img_size, None

    def close(self, folder):
        pass