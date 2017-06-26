import numpy as np


class BanditLearner():

    def __init__(self, num_of_bandits):
        self.bandits = [0] * num_of_bandits
        self.sums = [0] * num_of_bandits
        self.counts = [0] * num_of_bandits
        self.max_bandit = 0
        self.iteration_count = 0
        self.last_max_change = 0

    def _check_max_bandit(self, bandit, reward):
        if bandit != self.max_bandit:
            if self.bandits[bandit] > self.bandits[self.max_bandit]:
                self.max_bandit = bandit
                self.last_max_change = self.iteration_count
        elif reward < 0:
            max = np.argmax(self.bandits)
            if max != self.max_bandit:
                self.max_bandit = max
                self.last_max_change = self.iteration_count

    def give_reward(self, bandit, reward):
        self.sums[bandit] += reward
        self.counts[bandit] += 1
        self.bandits[bandit] = self.sums[bandit] / self.counts[bandit]
        self._check_max_bandit(bandit, reward)

    def give_reward_q_style(self, bandit, reward, discount_factor = 0.95, learning_factor=0.9):
        old_value = self.bandits[bandit]
        self.bandits[bandit] += learning_factor * (reward + discount_factor * old_value - old_value)
        self._check_max_bandit(bandit, -1)


    def choose_bandit(self, e=0.2, rand = False):
        if rand or np.random.rand(1) < e:
            bandit = np.random.randint(len(self.bandits))
        else:
            bandit = self.max_bandit
        return bandit

    def set_values(self, value):
        for i in range(len(self.bandits)):
            self.bandits[i] = value

    def increment_iteration_count(self):
        self.iteration_count += 1