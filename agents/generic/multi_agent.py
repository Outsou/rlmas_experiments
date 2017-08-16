from agents.generic.feature_agent import FeatureAgent
from creamas.math import gaus_pdf

import numpy as np
import aiomas
import logging


class MultiAgent(FeatureAgent):

    def __init__(self, environment, std, active=False, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.std = std
        self.sgd_reward = 0
        self.random_reward = 0
        self.max_reward = 0
        self.active = active

    def get_features(self, artifact):
        features = []
        for rule in self.R:
            features.append(rule.feat(artifact))
        return features

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)
        self.learner = MultiLearner(list(self.connections), len(self.R), self.std)
        return rets

    @aiomas.expose
    def give_artifact(self, artifact, eval, addr):
        features = self.get_features(artifact)
        self.learner.give_sample(eval, addr, features)

    @aiomas.expose
    async def act(self):
        artifact, _ = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
        features = self.get_features(artifact)
        eval, _ = self.evaluate(artifact)

        if not self.active:
            addr = list(self.connections.keys())[0]
            remote_agent = await self.env.connect(addr)
            await remote_agent.give_artifact(artifact, eval, self.addr)
        else:
            opinions = {}
            for addr in list(self.connections.keys()):
                remote_agent = await self.env.connect(addr)
                opinion, _ = await remote_agent.evaluate(artifact)
                opinions[addr] = opinion

            chosen_addr = self.learner.choose_addr(features, e=0.1)
            self.sgd_reward += opinions[chosen_addr]
            self.random_reward += np.sum(list(opinions.values())) / len(opinions)
            self.max_reward += opinions[max(opinions, key=opinions.get)]

            self.learner.give_sample(opinions[chosen_addr], chosen_addr, features)

    @aiomas.expose
    def close(self, folder=None):
        if not self.active:
            return

        self._log(logging.INFO, 'SGD reward: ' + str(self.sgd_reward))
        self._log(logging.INFO, 'Max reward: ' + str(self.max_reward))
        self._log(logging.INFO, 'Random reward: ' + str(self.random_reward))


class MultiLearner():
    def __init__(self, addrs, num_of_features, std, centroid_rate=200, weight_rate=0.2):
        self.centroid_rate = centroid_rate
        self.weight_rate = weight_rate
        self.num_of_features = num_of_features
        self.std = std

        self.sgd_weights = {}
        self.centroids = {}
        for addr in addrs:
            self.sgd_weights[addr] = np.array([0.5] * num_of_features)
            self.centroids[addr] = np.array([0.5] * num_of_features)

        self.max = gaus_pdf(1, 1, std)

    def _make_estimate(self, addr, features):
        vals = np.zeros(self.num_of_features)
        for i in range(self.num_of_features):
            vals[i] = gaus_pdf(features[i], self.centroids[addr][i], self.std) / self.max
        estimate = np.sum(self.sgd_weights[addr] * vals)
        return estimate, vals

    def give_sample(self, true_value, addr, features):
        estimate, vals = self._make_estimate(addr, features)

        error = true_value - estimate

        # Update weights
        gradient = vals * error
        self.sgd_weights[addr] += self.weight_rate * gradient
        self.sgd_weights[addr] = np.clip(self.sgd_weights[addr], 0, 1)

        # Calculate gradient of gaus pdf w.r.t mean
        gradient = (features - self.centroids[addr]) \
                   * np.exp(-(features - self.centroids[addr]) ** 2 / (2 * self.std ** 2)) \
                   / np.sqrt(2 * np.pi) * (self.std ** 2) ** (3 / 2)

        # Update centroid
        self.centroids[addr] += self.centroid_rate * gradient * error
        self.centroids[addr] = np.clip(self.centroids[addr], 0, 1)

    def choose_addr(self, features, e=0.2):
        if np.random.rand() < 0.2:
            return np.random.choice(list(self.sgd_weights.keys()))

        best_estimate = -1
        best_addr = None

        for addr in self.sgd_weights.keys():
            estimate, _ = self._make_estimate(addr, features)
            if estimate > best_estimate:
                best_estimate = estimate
                best_addr = addr

        return best_addr
