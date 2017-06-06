from agents.critic_test_agent import CriticTestAgent
from rl.bandit_learner import BanditLearner

import numpy as np
import logging
import aiomas


class CriticEqualAgent(CriticTestAgent):

    def __init__(self, environment, invent_n=120, *args, **kwargs):
        super().__init__(environment, invent_n, *args, **kwargs)
        self.name = self.addr
        self.creation_reward = 0
        self.criticism_reward = 0
        self.memory_states = (10, 60)
        self.memory_learner = BanditLearner(len(self.memory_states))
        self.current_memory_state = 0
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
        self.memory_learner.give_reward(self.current_memory_state,
                                        self.creation_reward + self.criticism_reward)
        self.creation_reward = 0
        self.criticism_reward = 0

    @aiomas.expose
    async def act(self):
        invent_n = self.invent_n/self.memory_states[self.current_memory_state]
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

class STMemory2():

    '''Agent's short-term memory model using a simple list which stores
    artifacts as is.'''
    def __init__(self, length):
        self.length = length
        self.artifacts = []

    def _add_artifact(self, artifact):
        # if len(self.artifacts) == self.length:
        #     self.artifacts = self.artifacts[:-1]
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

        if len(self.artifacts) < self.length:
            limit = len(self.artifacts)
        else:
            limit = self.length

        for i in range(limit):
            d = np.sqrt(np.sum(np.square(self.artifacts[i] - artifact)))
            if d < mdist:
                mdist = d
        return mdist