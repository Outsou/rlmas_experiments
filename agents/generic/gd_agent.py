from agents.generic.feature_agent import FeatureAgent

import numpy as np
import aiomas
import logging


class GDAgent(FeatureAgent):
    def __init__(self, environment, impressionability, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.total_reward = 0
        self.max_reward = 0
        self.expected_random_reward = 0
        self.age = 0
        self.chose_best = 0
        self.impressionability = impressionability
        self.recommendations = {'passed': [], 'rejected': []}

    @aiomas.expose
    def get_opinion(self, artifact):
        eval, _ = self.evaluate(artifact)
        if eval >= self._own_threshold:
            self.recommendations['passed'].append(eval)
            return 1
        self.recommendations['rejected'].append(eval)
        return 0

    @aiomas.expose
    def evaluate(self, artifact):
        if len(self.R) == 0:
            return 0.0, None

        s = 0
        w = 0.0
        if 'objective_values' in artifact.framings:
            objective_values = artifact.framings['objective_values']
        else:
            objective_values = np.zeros(len(self.R))
            for i in range(len(self.R)):
                value = self.R[i](artifact)
                objective_values[i] = value

        avg_weight = np.zeros(len(self.W))
        for addr, kwargs in self.connections.items():
            avg_weight += kwargs['weights']
        avg_weight /= len(self.W)

        weights = (1 - self.impressionability) * np.array(self.W) + self.impressionability * avg_weight

        for i in range(len(self.R)):
            s += objective_values[i] * weights[i]
            w += abs(weights[i])

        if w == 0.0:
            return 0.0, None
        return s / w, objective_values

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

        self.connection_counts = {}
        for conn in conns:
            self.connections[conn[0]]['weights'] = np.full(len(self.R), 0.5)

        return rets

    @aiomas.expose
    def get_recommendations(self):
        return self.recommendations

    @aiomas.expose
    def get_total_reward(self):
        return int(self.total_reward)

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
        artifact.framings['objective_values'] = values
        chosen_addr, estimate = choose_connection(values)

        opinions = {}
        for addr in self.connections:
            remote_agent = await self.env.connect(addr)
            opinion = await remote_agent.get_opinion(artifact)
            opinions[addr] = opinion

        max_reward = opinions[max(opinions, key=opinions.get)]
        self.max_reward += max_reward
        #self.total_reward += opinions[chosen_addr]
        self.total_reward += np.sum(list(opinions.values()))
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
        passed_len = len(self.recommendations['passed'])
        if passed_len > 0:
            self._log(logging.INFO, '{}/{} recommendations passed with avg evaluation {}'
                      .format(passed_len,
                              passed_len + len(self.recommendations['rejected']),
                              np.sum(self.recommendations['passed']) / passed_len))
        else:
            self._log(logging.INFO, '0 recommendations passed')
        #self._log(logging.INFO, str(self.connections))
