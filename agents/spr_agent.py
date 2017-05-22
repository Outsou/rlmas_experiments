from creamas.examples.spiro.spiro_agent import SpiroAgent
from rl.q_learner import QLearner
import logging
import aiomas
import numpy as np


class SprAgent(SpiroAgent):

    def __init__(self, environment, states, rand = False, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.states = states
        self.actions = []
        self.learner = None
        self.rand = rand
        self.total_reward = 0

    @aiomas.expose
    def set_acquaintances(self, addresses):
        addresses = list(addresses)
        addresses.remove(self.addr)

        self.acquaintances = {}

        for acquaintance in addresses:
            self.acquaintances[acquaintance] = 0
            for state in self.states:
                self.actions.append((state, acquaintance))

        self.learner = QLearner(len(self.states), len(self.actions))
        self.learner.set_initial_state(0)

    @aiomas.expose
    def get_total_reward(self):
        return self.total_reward

    @aiomas.expose
    def log_situation(self):
        self._log(logging.INFO, 'Random: {}'.format(self.rand))
        self._log(logging.INFO, 'Reward: {}'.format(self.last_reward))
        self._log(logging.INFO, 'Total reward: {}'.format(self.total_reward))

    @aiomas.expose
    async def ask_opinion(self, artifact):
        evaluation, _ = self.evaluate(artifact)
        #print(evaluation)
        self.learn(artifact)
        return evaluation

    @aiomas.expose
    async def act(self):
        # Select action randomly, or use Q-learning
        if self.rand:
            action = np.random.randint(len(self.actions))
        else:
            action = self.learner.choose_action(1)

        # Create new artifact based on selected state
        state = self.actions[action][0]
        self.spiro_args = np.array(state)
        artifact = self.invent(10)

        # Keep track of selected acquaintances
        self.acquaintances[self.actions[action][1]] += 1

        # Get evaluation
        chosen_acquaintance = await self.env.connect(self.actions[action][1])
        reward = await chosen_acquaintance.ask_opinion(artifact)
        self.last_reward = reward
        self.total_reward += reward

        # Learn from the evaluation
        self.learner.give_reward(self.states.index(state), reward)

        self.learn(artifact)

    @aiomas.expose
    def close(self, folder=None):
        self._log(logging.INFO, self.acquaintances)



