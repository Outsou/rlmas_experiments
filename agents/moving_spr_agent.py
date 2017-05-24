from creamas.examples.spiro.spiro_agent import SpiroAgent
from creamas.examples.spiro.spiro_agent import SpiroArtifact
from rl.q_learner import QLearner
from rl.q_learner import QLearner
import logging

import aiomas
import numpy as np


class MovingSprAgent(SpiroAgent):
    '''
    state: last movement dir & was last artifact passed or rejected
    action: movement dir
    '''

    def __init__(self, environment, step_size=1, start_location = np.array([0, 0]), *args, **kwargs):
        super().__init__(environment, *args, **kwargs)

        self.spiro_args = start_location
        self.step_size = step_size

        self.actions = ('N', 'E', 'W', 'S')
        last_novel = ('reject', 'pass')
        self.states = []

        for last_direction in self.actions:
            for novelty in last_novel:
                self.states.append((last_direction, novelty))

        self.learner = QLearner(len(self.states) + 1, len(self.actions), initial_values=1)
        self.learner.set_initial_state(len(self.states))

        self.arg_history.append(self.spiro_args)
        self.total_reward = 0

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

    @aiomas.expose
    async def act(self):

        #action = self.actions[self.learner.choose_action_e_greedy()]
        action = self.actions[self.learner.choose_action_softmax(0.5)]
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

        new_state = (action, artifact.self_criticism)
        self.learner.give_reward(self.states.index(new_state), val)

        self.total_reward += val

    def close(self, folder):
        self.plot_places()