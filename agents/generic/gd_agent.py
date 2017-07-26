from agents.generic.feature_agent import FeatureAgent

import numpy as np
import aiomas
import logging


class GDAgent(FeatureAgent):
    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.total_reward = 0
        self.max_reward = 0
        self.random_reward = 0

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

        artifact = self.invent(self.search_width)
        eval, values = self.evaluate(artifact)
        chosen_addr, estimate = choose_connection(values)

        opinions = {}
        for addr in self.connections:
            opinion, _ = await self.ask_opinion(addr, artifact)
            opinions[addr] = opinion

        self.max_reward += opinions[max(opinions, key=opinions.get)]
        self.total_reward += opinions[chosen_addr]
        self.random_reward += opinions[np.random.choice(list(opinions.keys()))]
        gradient =(estimate - opinion) * values
        self.connections[chosen_addr]['weights'] -= gradient

    @aiomas.expose
    def close(self, folder=None):
        self._log(logging.INFO, '-------------------------------------------------------')
        self._log(logging.INFO, 'Total reward: ' + str(self.total_reward))
        self._log(logging.INFO, 'Max reward: ' + str(self.max_reward))
        self._log(logging.INFO, 'Random reward: ' + str(self.random_reward))
