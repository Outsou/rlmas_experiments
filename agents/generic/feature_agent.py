from creamas.rules.agent import RuleAgent
from creamas.math import gaus_pdf
from rl.bandit_learner import BanditLearner

import logging
import aiomas
import numpy as np


class FeatureAgent(RuleAgent):
    def __init__(self, environment, artifact_cls, create_kwargs, desired_novelty=-1,
                 ask_criticism=True, search_width=10, ask_random=False,
                 critic_threshold=10, veto_threshold=10, log_folder=None,
                 log_level=logging.INFO, memsize=100, hedonic_std=0.1):
        super().__init__(environment, log_folder=log_folder,
                         log_level=log_level)

        self.stmem = STMemory(artifact_cls=artifact_cls, length=memsize, max_length=memsize)
        self.artifact_cls = artifact_cls
        self.age = 0
        self._own_threshold = critic_threshold
        self._novelty_threshold = veto_threshold
        self.search_width = search_width
        self.ask_random = ask_random
        self.ask_criticism = ask_criticism
        self.desired_novelty = desired_novelty
        self.hedonic_std = hedonic_std
        self.create_kwargs = create_kwargs

        self.bandit_learner = None
        self.connection_counts = None

        self.comparison_count = 0
        self.artifacts_created = 0
        self.passed_self_criticism_count = 0

    def novelty(self, artifact):
        self.comparison_count += self.stmem.get_comparison_amount()
        distance = self.stmem.distance(artifact)
        return distance

    def hedonic_value(self, value, desired_value):
        lmax = gaus_pdf(desired_value, desired_value, self.hedonic_std)
        pdf = gaus_pdf(value, desired_value, self.hedonic_std)
        return pdf / lmax

    def evaluate(self, artifact):
        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        novelty = self.novelty(artifact)

        if self.desired_novelty > 0:
            evaluation = self.hedonic_value(novelty, self.desired_novelty)
        else:
            evaluation = novelty

        artifact.add_eval(self, evaluation)

        return evaluation, None

    def invent(self, n):
        best_artifact = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
        return best_artifact

    def learn(self, artifact, iterations=1):
        '''Train short term memory with given spirograph.
        :param spiro:
            :py:class:`SpiroArtifact` object
        '''
        for i in range(iterations):
            self.stmem.train_cycle(artifact)

    @aiomas.expose
    def get_addr(self):
        return self.addr

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)
        self.gatekeepers = list(self._connections.keys())
        length = len(self.gatekeepers)
        self.bandit_learner = BanditLearner(length)

        self.connection_counts = {}
        for conn in conns:
            self.connection_counts[conn[0]] = 0

        return rets

    @aiomas.expose
    async def get_criticism(self, artifact):
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._novelty_threshold:
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    def get_connection_counts(self):
        return self.connection_counts

    @aiomas.expose
    def get_comparison_count(self):
        return self.comparison_count

    @aiomas.expose
    def get_artifacts_created(self):
        return self.artifacts_created

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    def get_desired_novelty(self):
        return self.desired_novelty

    @aiomas.expose
    def get_passed_self_criticism_count(self):
        return self.passed_self_criticism_count

    @aiomas.expose
    async def act(self):
        art = self.invent(10)
        # self.age += 1
        # self.bandit_learner.increment_iteration_count()
        #
        # artifact = self.invent(self.search_width)
        #
        # val = artifact.evals[self.name]
        # self.add_artifact(artifact)
        #
        # if val >= self._own_threshold:
        #     artifact.self_criticism = 'pass'
        #     self.passed_self_criticism_count += 1
        #     self.learn(artifact)
        #
        #     if not self.ask_criticism:
        #         self.add_candidate(artifact)
        #         self.added_last = True
        #         return
        #
        #     bandit = self.bandit_learner.choose_bandit(rand=self.ask_random)
        #     critic = self.gatekeepers[bandit]
        #     self.connection_counts[critic] += 1
        #
        #     connection = await self.env.connect(critic)
        #     passed, artifact = await connection.get_criticism(artifact)
        #
        #     if passed:
        #         self.bandit_learner.give_reward(bandit, -1)
        #         self.add_candidate(artifact)
        #         self.added_last = True
        #     else:
        #         self.bandit_learner.give_reward(bandit, 1)


class STMemory:

    '''Agent's short-term memory model using a simple list which stores
    artifacts as is.'''
    def __init__(self, artifact_cls, length, max_length = 100):
        self.length = length
        self.artifacts = []
        self.max_length = max_length
        self.artifact_cls = artifact_cls

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
        limit = self.get_comparison_amount()
        mdist = self.artifact_cls.max_distance(artifact)
        if limit == 0:
            return np.random.random() * mdist
        for a in self.artifacts[:limit]:
            d = self.artifact_cls.distance(artifact, a)
            if d < mdist:
                mdist = d
        return mdist

    def get_comparison_amount(self):
        if len(self.artifacts) < self.length:
            amount = len(self.artifacts)
        else:
            amount = self.length
        return amount
