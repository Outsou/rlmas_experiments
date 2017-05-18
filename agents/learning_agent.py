from creamas.grid import GridAgent
import aiomas
import numpy as np

class LearningAgent(GridAgent):

    def __init__(self, environment, *args, **kwargs):
        self.learner = kwargs.pop('learner')
        self.learner.set_initial_state(0)

        self.difficulty_preference = kwargs.pop('difficulty_preference')
        super().__init__(environment, *args, **kwargs)
        self.actions = []

    def map_actions_to_neighbors(self):
        for card, neighbor in self.neighbors.items():
            if neighbor is not None:
                self.actions.append(neighbor)

    @aiomas.expose
    async def get_number(self):
        return self.difficulty_preference

    @aiomas.expose
    async def act(self, *args, **kwargs):
        action = self.learner.choose_action(1)
        neighbor = await self.env.connect(self.actions[action])
        reward = np.exp(-abs(self.difficulty_preference - await neighbor.get_number()))
        self.learner.give_reward(0, reward)
