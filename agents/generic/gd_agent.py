from agents.generic.feature_agent import FeatureAgent

import numpy as np
import aiomas
import logging


class GDAgent(FeatureAgent):
    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.total_reward = 0
        self.max_reward = 0
        self.expected_random_reward = 0
        self.age = 0
        self.chose_best = 0

    @aiomas.expose
    def evaluate(self, artifact):
        s = 0
        w = 0.0
        rule_values = np.zeros(len(self.R))

        if len(self.R) == 0:
            return 0.0, None

        for i in range(len(self.R)):
            value = self.R[i](artifact)
            rule_values[i] = value
            s += value * self.W[i]
            w += abs(self.W[i])

        if w == 0.0:
            return 0.0, None
        return s / w, rule_values

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

        self.connection_counts = {}
        for conn in conns:
            self.connections[conn[0]]['weights'] = np.full(len(self.R), 0.5)

        return rets

    @aiomas.expose
    async def act(self):
        def choose_connection(values):
            if np.random.random() < 0.2:
                addr = np.random.choice(list(self.connections.keys()))
                estimate = np.sum(values * self.connections[addr]['weights'])
                return addr, estimate

            estimates = {}
            for addr in self.connections.keys():
                estimates[addr] = np.sum(values * self.connections[addr]['weights'])

            max_addr = max(estimates, key=estimates.get)
            return max_addr, estimates[max_addr]

        self.age += 1
        artifact = self.invent(self.search_width)
        eval, values = self.evaluate(artifact)
        chosen_addr, estimate = choose_connection(values)

        opinions = {}
        for addr in self.connections:
            opinion, _ = await self.ask_opinion(addr, artifact)
            opinions[addr] = opinion

        max_reward = opinions[max(opinions, key=opinions.get)]
        self.max_reward += max_reward
        self.total_reward += opinions[chosen_addr]
        self.expected_random_reward += np.sum(list(opinions.values())) / len(opinions)

        if max_reward == opinions[chosen_addr]:
            self.chose_best += 1

        gradient =(estimate - opinions[chosen_addr]) * values
        self.connections[chosen_addr]['weights'] -= gradient

    @aiomas.expose
    def close(self, folder=None):
        self._log(logging.INFO, '-------------------------------------------------------')
        self._log(logging.INFO, 'Total reward: ' + str(self.total_reward))
        self._log(logging.INFO, 'Max reward: ' + str(self.max_reward))
        self._log(logging.INFO, 'Expected random reward: ' + str(self.expected_random_reward))
        self._log(logging.INFO, 'Chose best {}/{} times'.format(self.chose_best, self.age))
        #self._log(logging.INFO, str(self.connections))
