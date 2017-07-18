from agents.generic.feature_agent import FeatureAgent
from rl.bandit_learner import BanditLearner

import aiomas
import numpy as np
import logging


class CreatorAgent(FeatureAgent):

    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.rule = None
        self.connection_list = []
        self.bandits = []
        self.total_reward = 0
        self.max_reward = 0
        self.random_reward = 0
        self.chose_max_count = 0
        self.age = 0

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)
        self.connection_list = list(self._connections.keys())

        for _ in range(len(self.R)):
            bandit = BanditLearner(len(self.connection_list))
            bandit.set_values(1)
            self.bandits.append(bandit)

        return rets

    def evaluate(self, artifact):
        return self.rule(artifact), None

    @aiomas.expose
    async def act(self):
        self.age += 1
        self.rule = np.random.choice(self.R)
        artifact = self.invent(self.search_width)

        max_idx = -1
        max_value = -1

        for i in range(len(self.R)):
            value = self.R[i](artifact)
            if value > max_value:
                max_value = value
                max_idx = i

        evals = []

        for addr in self.connection_list:
            eval, _ = await self.ask_opinion(addr, artifact)
            evals.append(eval)

        chosen_critic = self.bandits[max_idx].choose_bandit()
        reward = evals[chosen_critic]
        self.bandits[max_idx].give_reward(chosen_critic, reward)

        max_reward = max(evals)
        if reward == max_reward:
            self.chose_max_count += 1

        self.total_reward += reward
        self.max_reward += max_reward
        self.random_reward += np.random.choice(evals)

    @aiomas.expose
    def close(self, folder=None):
        self._log(logging.INFO, 'Total reward: ' + str(self.total_reward))
        self._log(logging.INFO, 'Max reward: ' + str(self.max_reward))
        self._log(logging.INFO, 'Random reward: ' + str(self.random_reward))
        self._log(logging.INFO, 'Chose maximum critic {}/{} times: '.format(self.chose_max_count, self.age))
