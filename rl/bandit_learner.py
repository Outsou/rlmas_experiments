import numpy as np


class BanditLearner():

    def __init__(self, num_of_bandits):
        self.bandits = [0] * num_of_bandits
        self.sums = [0] * num_of_bandits
        self.counts = [0] * num_of_bandits
        self.max_bandit = 0
        self.iteration_count = 0
        self.last_max_change = 0

    def give_reward(self, bandit, reward):
        self.iteration_count += 1
        self.sums[bandit] += reward
        self.counts[bandit] += 1
        self.bandits[bandit] = self.sums[bandit] / self.counts[bandit]

        if bandit != self.max_bandit:
            if self.bandits[bandit] > self.bandits[self.max_bandit]:
                self.max_bandit = bandit
                self.last_max_change = self.iteration_count
        elif reward < 0:
            max = np.argmax(self.bandits)
            if max != self.max_bandit:
                self.max_bandit = max
                self.last_max_change = self.iteration_count

    def choose_bandit(self, e=0.2, rand = False):
        if rand or np.random.rand(1) < e:
            bandit = np.random.randint(len(self.bandits))
        else:
            bandit = self.max_bandit
        return bandit