from creamas.examples.spiro.spiro_agent_mp import SpiroAgent
from rl.q_learner import QLearner


class SprAgent(SpiroAgent):

    def __init__(self, environment, states, *args, **kwargs):
        super().__init__(environment, 2, *args, **kwargs)
        self.states = states

    def set_acquaintances(self):
        addresses = self.env.get_agents()
        addresses.remove(self.addr)

        self.actions = []

        for state in self.states:
            for acquaintance in addresses:
                self.actions.append((state, acquaintance))

        self.learner = QLearner(len(self.states), len(self.actions))
