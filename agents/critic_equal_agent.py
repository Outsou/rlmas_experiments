from agents.critic_test_agent import CriticTestAgent
from rl.bandit_learner import BanditLearner

import numpy as np
import logging
import aiomas


class CriticEqualAgent(CriticTestAgent):

    def __init__(self, environment, memory_states, initial_state = 0, invent_n=120, *args, **kwargs):
        super().__init__(environment, invent_n, *args, **kwargs)
        self.name = self.addr
        self.creation_reward = 0
        self.criticism_reward = 0
        self.memory_states = memory_states
        self.memory_state_times = [0] * len(self.memory_states)
        self.memory_learner = BanditLearner(len(self.memory_states))
        self.current_memory_state = initial_state
        self.stmem = STMemory2(self.memory_states[0])
        self.invent_n = invent_n

    @aiomas.expose
    def ask_if_passes(self, artifact):
        passes, artifact = super().ask_if_passes(artifact)

        if not passes:
            self.criticism_reward += 1

        return passes, artifact

    @aiomas.expose
    def process_rewards(self):
        '''Called after voting, so agent can process all the reward gained at once'''
        # Record in which state this time step was spent
        self.memory_state_times[self.current_memory_state] += 1

        # Reward is the sum of creation and criticism rewards of this time step
        self.memory_learner.give_reward(self.current_memory_state,
                                        self.creation_reward + self.criticism_reward)

        # Set rewards to zero for next time step
        self.creation_reward = 0
        self.criticism_reward = 0

        # Choose next memory state
        self.current_memory_state = self.memory_learner.choose_bandit()
        self.stmem.length = self.memory_states[self.current_memory_state]

    @aiomas.expose
    async def act(self):
        self.bandit_learner.increment_iteration_count()

        # Invent artifact using constant amount of comparisons
        invent_n = self.invent_n / self.memory_states[self.current_memory_state]
        artifact = self.invent(int(invent_n))

        self.added_last = False

        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self.spiro_args = args
        self.arg_history.append(self.spiro_args)
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            self.overcame_own_threshold_count += 1
            artifact.self_criticism = 'pass'
            # Train SOM with the invented artifact
            self.learn(artifact, self.teaching_iterations)

            # Check with another agent if the artifact is novel enough
            if self.ask_passing:
                # Ask someone for veto
                bandit = self.bandit_learner.choose_bandit(rand=self.rand)
                acquaintance = self.acquaintances[bandit]
                acquaintance[1] += 1

                connection = await self.env.connect(acquaintance[0])
                passed, artifact = await connection.ask_if_passes(artifact)

                if passed:
                    self.bandit_learner.give_reward(bandit, -1)
                    self.env.add_candidate(artifact)
                    self.added_last = True
                else:
                    self.bandit_learner.give_reward(bandit, 1)
            else:
                self.env.add_candidate(artifact)
                self.added_last = True
        elif self.jump == 'random':
            largs = self.spiro_args
            self.spiro_args = np.random.uniform(-199, 199,
                                                self.spiro_args.shape)
            self._log(logging.DEBUG, "Jumped from {} to {}"
                      .format(largs, self.spiro_args))

    @aiomas.expose
    async def domain_artifact_added(self, spiro, iterations=1):
        if spiro.creator == self.name:
            self.creation_reward += 1

    def evaluate(self, artifact):
        '''Evaluate the artifact with respect to the agents short term memory.

        Returns value in [0, 1].
        '''

        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        # Keep track of comparisons
        self.comparison_count += self.stmem.get_comparison_amount()

        if self.desired_novelty > 0:
            return self.hedonic_value(self.novelty(artifact.obj))
        return self.novelty(artifact.obj) / self.img_size, None

    @aiomas.expose
    def get_memory_state_times(self):
        return self.memory_state_times

class STMemory2():

    '''Agent's short-term memory model using a simple list which stores
    artifacts as is.'''
    def __init__(self, length, max_length = 100):
        self.length = length
        self.artifacts = []
        self.max_length = max_length

    def _add_artifact(self, artifact):
        if len(self.artifacts) >= 2 * self.max_length:
            self.artifacts = self.artifacts[:self.max_length]
        self.artifacts.insert(0, artifact)

    def learn(self, artifact):
        '''Learn new artifact. Removes last artifact from the memory if it is
        full.'''
        self._add_artifact(artifact)

    def train_cycle(self, artifact):
        '''Train cycle method to keep the interfaces the same with the SOM
        implementation of the short term memory.
        '''
        self.learn(artifact)

    def distance(self, artifact):
        mdist = np.sqrt(artifact.shape[0])
        if len(self.artifacts) == 0:
            return np.random.random()*mdist

        limit = self.get_comparison_amount()

        for i in range(limit):
            d = np.sqrt(np.sum(np.square(self.artifacts[i] - artifact)))
            if d < mdist:
                mdist = d
        return mdist

    def get_comparison_amount(self):
        if len(self.artifacts) < self.length:
            amount = len(self.artifacts)
        else:
            amount = self.length
        return amount