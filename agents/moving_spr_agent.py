from creamas.examples.spiro.spiro_agent import SpiroAgent
from creamas.examples.spiro.spiro_agent import SpiroArtifact
from rl.q_learner import QLearner
from rl.q_learner import QLearner
import logging

import aiomas
import numpy as np


class QMovingSprAgent(SpiroAgent):
    '''
    state: last movement dir & was last artifact passed or rejected
    action: movement dir
    '''

    def __init__(self, environment, step_size=1, start_location = np.array([0, 0]), initial_values=0, discount_factor=0.85, learning_factor=0.8, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)

        self.spiro_args = start_location
        self.step_size = step_size

        self.actions = ('N', 'E', 'W', 'S')
        self.previous_action = np.random.choice(self.actions)
        reward_directions = ('down', 'up')
        action_pairs = [(x, y) for x in self.actions for y in self.actions]

        self.states = [action_pair + (reward_direction, ) for action_pair in action_pairs for reward_direction in reward_directions]

        self.learner = QLearner(len(self.states) + 1, len(self.actions), initial_values=initial_values, discount_factor=discount_factor, learning_factor=learning_factor)
        self.learner.set_initial_state(len(self.states))

        self.arg_history.append(self.spiro_args)
        self.total_reward = 0

        self.last_reward = 0

    def invent(self, n):
        '''Invent new spirograph by taking n random steps from current position
        (spirograph generation parameters) and selecting the best one based
        on the agent's evaluation (hedonic function).
        :param int n: how many spirographs are created for evaluation
        :returns: Best created artifact.
        :rtype: :py:class:`~creamas.core.agent.Artifact`
        '''
        args = self.randomize_args()
        img = self.create(args[0], args[1])
        best_artifact = SpiroArtifact(self, img, domain='image')
        ev, _ = self.evaluate(best_artifact)
        best_artifact.add_eval(self, ev, fr={'args': args})
        for i in range(n-1):
            args = self.randomize_args()
            img = self.create(args[0], args[1])
            artifact = SpiroArtifact(self, img, domain='image')
            ev, _ = self.evaluate(artifact)
            artifact.add_eval(self, ev, fr={'args': args})
            if ev > best_artifact.evals[self.name]:
                best_artifact = artifact
        best_artifact.in_domain = False
        best_artifact.self_criticism = 'reject'
        best_artifact.creation_time = self.age
        return best_artifact

    async def act(self):

        #action = self.actions[self.learner.choose_action_e_greedy()]
        action = self.actions[self.learner.choose_action_softmax(1.5)]
        assert action in self.actions

        step = self.step_size

        if action is 'N':
            self.spiro_args[1] += step
        elif action is 'S':
            self.spiro_args[1] -= step
        elif action is 'W':
            self.spiro_args[0] -= step
        elif action is 'E':
            self.spiro_args[0] += step

        artifact = self.invent(self.search_width)
        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self._log(logging.INFO, "Created spirograph with args={}, val={}".format(args, val))
        self.arg_history.append(args)

        self.add_artifact(artifact)
        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.learn(artifact, self.teaching_iterations)

        if val >= self.last_reward:
            reward_direction = 'up'
        else:
            reward_direction = 'down'

        new_state = (self.previous_action, action, reward_direction)
        self.learner.give_reward(self.states.index(new_state), val)
        self.previous_action = action
        self.total_reward += val

    def close(self, folder):
        self.plot_places()

class BasicMovingSprAgent(SpiroAgent):

    def __init__(self, environment, start_location = np.array([0, 0]), initial_values=0, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.spiro_args = start_location
        self.total_reward = 0

    async def act(self):
        '''Agent's main method to create new spirographs.
        See Simulation and CreativeAgent documentation for details.
        '''

        # Invent new artifact
        artifact = self.invent(self.search_width)
        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self.spiro_args = args
        self.arg_history.append(self.spiro_args)
        self.add_artifact(artifact)
        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.learn(artifact, self.teaching_iterations)
        elif self.jump == 'random':
            self.spiro_args = np.random.uniform(-199, 199,
                                                self.spiro_args.shape)

        self.total_reward += val

    def close(self, folder):
        pass