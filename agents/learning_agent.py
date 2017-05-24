from creamas.grid import GridAgent
import aiomas
import numpy as np

class LearningAgent(GridAgent):

    def __init__(self, environment, standard_deviation = 1, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.learner = kwargs.pop('learner')
        self.standard_deviation = standard_deviation
        self.learner.set_initial_state(0)

        self.difficulty_preference = kwargs.pop('difficulty_preference')
        self.actions = []

    def map_actions_to_neighbors(self):
        for card, neighbor in self.neighbors.items():
            if neighbor is not None:
                self.actions.append(neighbor)

    @aiomas.expose
    async def get_number(self):
        number = np.random.normal(self.difficulty_preference, self.standard_deviation)
        #print("Preference; {}, generated: {}".format(self.difficulty_preference, number))
        return number

    @aiomas.expose
    async def act(self, *args, **kwargs):
        action = self.learner.choose_action_softmax(1)
        neighbor = await self.env.connect(self.actions[action])
        reward = np.exp(-abs(self.difficulty_preference - await neighbor.get_number()))
        self.learner.give_reward(0, reward)
