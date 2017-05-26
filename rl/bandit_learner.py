import numpy as np


class BanditLearner():

    def __init__(self, num_of_bandits):
        self.bandits = [0] * num_of_bandits
        self.sums = [0] * num_of_bandits
        self.counts = [0] * num_of_bandits

    def give_reward(self, bandit, reward):
        self.sums[bandit] += reward
        self.counts[bandit] += 1
        self.bandits[bandit] = self.sums[bandit] / self.counts[bandit]

    def choose_bandit(self, e=0.2):
        if np.random.rand(1) < e:
            bandit = np.random.randint(len(self.bandits))
        else:
            bandit = np.argmax(self.bandits)
        return bandit