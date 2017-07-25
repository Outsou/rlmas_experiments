from creamas.rules.agent import RuleAgent
from creamas.math import gaus_pdf
from rl.bandit_learner import BanditLearner

import logging
import aiomas
import numpy as np
from matplotlib import pyplot as plt


class FeatureAgent(RuleAgent):
    def __init__(self, environment, artifact_cls, create_kwargs, rules, rule_weights = None, desired_novelty=-1,
                 novelty_weight=0.5, ask_criticism=True, search_width=10, ask_random=False,
                 critic_threshold=0.5, veto_threshold=0.5, log_folder=None,
                 log_level=logging.INFO, memsize=0, hedonic_std=0.1):
        super().__init__(environment, log_folder=log_folder,
                         log_level=log_level)

        max_distance = artifact_cls.max_distance(create_kwargs)
        self.stmem = STMemory(artifact_cls=artifact_cls, length=memsize, max_length=memsize, max_distance=max_distance)
        self.artifact_cls = artifact_cls
        self._own_threshold = critic_threshold
        self._novelty_threshold = veto_threshold
        self.search_width = search_width
        self.ask_random = ask_random
        self.ask_criticism = ask_criticism
        self.desired_novelty = desired_novelty
        self.hedonic_std = hedonic_std
        self.create_kwargs = create_kwargs
        self.novelty_weight = novelty_weight

        self.connection_counts = None
        self.connection_list = []

        self.comparison_count = 0
        self.artifacts_created = 0
        self.passed_self_criticism_count = 0

        if rule_weights is None:
            rule_weights = [1] * len(rules)
        else:
            assert len(rules) == len(rule_weights), "Different amount of rules and rule weights."

        for i in range(len(rules)):
            self.add_rule(rules[i], rule_weights[i])

    def novelty(self, artifact):
        self.comparison_count += self.stmem.get_comparison_amount()
        distance = self.stmem.distance(artifact)
        return distance

    def hedonic_value(self, value, desired_value):
        lmax = gaus_pdf(desired_value, desired_value, self.hedonic_std)
        pdf = gaus_pdf(value, desired_value, self.hedonic_std)
        return pdf / lmax

    @aiomas.expose
    def evaluate(self, artifact):
        if self.name in artifact.evals:
            return artifact.evals[self.name], artifact.framings['eval_values']

        value, _ = super().evaluate(artifact)
        novelty = self.novelty(artifact)
        evaluation = (1 - self.novelty_weight) * value + self.novelty_weight * novelty

        artifact.add_eval(self, evaluation)
        eval_values = {'value': value, 'novelty': novelty}
        artifact.framings['eval_values'] = eval_values

        return evaluation, eval_values

    def invent(self, n):
        best_artifact, _ = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
        return best_artifact

    def learn(self, artifact, iterations=1):
        for i in range(iterations):
            self.stmem.train_cycle(artifact)

    @aiomas.expose
    def get_addr(self):
        return self.addr

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

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
        artifact = self.invent(self.search_width)
        eval, eval_values = self.evaluate(artifact)
        artifact.framings['eval_values'] = eval_values
        self._log(logging.INFO, 'Created artifact with evaluation {} (value: {}, novelty: {})'
                  .format(eval, eval_values['value'], eval_values['novelty']))
        self.add_artifact(artifact)

        if eval >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.passed_self_criticism_count += 1
            if self.stmem.length > 0 and eval_values['novelty'] > 0:
                self.learn(artifact)

    @aiomas.expose
    def save_artifacts(self, folder):
        i = 0
        for art in reversed(self.stmem.artifacts[:self.stmem.length]):
            i += 1
            plt.imshow(art.obj, shape=art.obj.shape)
            plt.title('Eval: {}'.format(art.evals[self.name]))
            plt.savefig('{}/artifact{}'.format(folder, i))


class STMemory:

    '''Agent's short-term memory model using a simple list which stores
    artifacts as is.'''
    def __init__(self, artifact_cls, length, max_distance, max_length = 100):
        self.length = length
        self.artifacts = []
        self.max_length = max_length
        self.artifact_cls = artifact_cls
        self.max_distance = max_distance

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
        if limit == 0:
            return np.random.random() * self.max_distance / self.max_distance
        min_distance = self.max_distance
        for a in self.artifacts[:limit]:
            d = self.artifact_cls.distance(artifact, a)
            if d < min_distance:
                min_distance = d
        return min_distance / self.max_distance

    def get_comparison_amount(self):
        if len(self.artifacts) < self.length:
            amount = len(self.artifacts)
        else:
            amount = self.length
        return amount
