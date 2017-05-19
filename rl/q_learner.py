import numpy as np
from utilities import math


class QLearner:
    def __init__(self, states, actions, discount_factor = 0.8, learning_factor = 0.85):
        self.q_table = np.zeros((states, actions))
        self.actions = actions
        self.learning_factor = learning_factor
        self.discount_factor = discount_factor
        self.current_state = None
        self.action = None

    def choose_action(self, temperature):
        action_probs = math.softmax(self.q_table[self.current_state], temperature)
        self.action = np.random.choice(self.actions, p=action_probs)
        return self.action

    def give_reward(self, new_state, reward):
        change = self.learning_factor * (reward + self.discount_factor * np.max(self.q_table[new_state, :]) - self.q_table[self.current_state, self.action])
        self.q_table[self.current_state, self.action] += change
        self.current_state = new_state

    def set_initial_state(self, state):
        self.current_state = state
