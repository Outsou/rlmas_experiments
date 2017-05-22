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

        for state in self.states:
            for acquaintance in addresses:
                self.actions.append((state, acquaintance))

        self.learner = QLearner(len(self.states), len(self.actions))
        self.learner.set_initial_state(0)

    @aiomas.expose
    async def ask_opinion(self, artifact):
        evaluation, _ = self.evaluate(artifact)
        #print(evaluation)
        self.learn(artifact)
        return evaluation

    @aiomas.expose
    async def act(self):
        if self.rand:
            action = np.random.randint(len(self.actions))
        else:
            action = self.learner.choose_action(1)

        state = self.actions[action][0]
        self.spiro_args = np.array(state)
        artifact = self.invent(10)

        chosen_acquaintance = await self.env.connect(self.actions[action][1])
        reward = await chosen_acquaintance.ask_opinion(artifact)
        self.total_reward += reward
        self.learner.give_reward(self.states.index(state), reward)

        self.learn(artifact)

        self._log(logging.INFO, 'Random: {}'.format(self.rand))
        self._log(logging.INFO, 'Reward: {}'.format(reward))
        self._log(logging.INFO, 'Total reward: {}'.format(self.total_reward))


    @aiomas.expose
    def close(self, folder=None):
        pass



