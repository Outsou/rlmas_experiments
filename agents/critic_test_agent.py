from creamas.examples.spiro.spiro_agent_mp import SpiroAgent
from rl.bandit_learner import BanditLearner
from artifacts.spr_artifact import SpiroArtifact

import aiomas
import numpy as np
import logging


class CriticTestAgent(SpiroAgent):

    def __init__(self, environment, ask_passing=True, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.comparison_count = 0
        self.name = "{}_M{}".format(self.name, self.stmem.length)
        self.validated_something = False
        self.ask_passing = ask_passing

    @aiomas.expose
    def set_acquaintances(self, addresses):
        addresses = list(addresses)
        addresses.remove(self.addr)

        self.acquaintances = []

        for acquaintance in addresses:
            self.acquaintances.append([acquaintance, 0])

        self.bandit_learner = BanditLearner(len(self.acquaintances))


    @aiomas.expose
    def ask_if_passes(self, artifact):
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._novelty_threshold:
            artifact.add_eval(self, evaluation)
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    async def act(self):
        artifact = self.invent(10)

        self.added_last = False

        args = artifact.framings[self.name]['args']
        val = artifact.evals[self.name]
        self.spiro_args = args
        self.arg_history.append(self.spiro_args)
        self.add_artifact(artifact)

        if val >= self._own_threshold:
            artifact.self_criticism = 'pass'
            # Train SOM with the invented artifact
            self.learn(artifact, self.teaching_iterations)

            # Check with another agent if the artifact is novel enough
            if self.ask_passing:
                # Ask someone for veto
                bandit = self.bandit_learner.choose_bandit()
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
    def get_addr(self):
        return self.addr

    @aiomas.expose
    def get_acquaintances(self):
        return self.acquaintances

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    def get_comparison_count(self):
        return self.comparison_count

    @aiomas.expose
    def get_acquaintance_values(self):
        acquaintance_values = {}

        for i in range(len(self.acquaintances)):
            acquaintance_values[self.acquaintances[i][0]] = self.bandit_learner.bandits[i]

        return acquaintance_values

    @aiomas.expose
    def get_last_best_acquaintance_change(self):
        return self.bandit_learner.last_max_change

    def evaluate(self, artifact):
        '''Evaluate the artifact with respect to the agents short term memory.

        Returns value in [0, 1].
        '''

        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        # Keep track of comparisons
        self.comparison_count += len(self.stmem.artifacts)

        if self.desired_novelty > 0:
            return self.hedonic_value(self.novelty(artifact.obj))
        return self.novelty(artifact.obj) / self.img_size, None

    def close(self, folder):
        pass

    def validate(self, candidates):
        besteval = 0.0
        bestcand = None
        valid = []
        for c in candidates:
            if c.creator != self.name:
                ceval, _ = self.evaluate(c)
                if ceval >= self._novelty_threshold:
                    valid.append(c)
                    if ceval > besteval:
                        besteval = ceval
                        bestcand = c
            else:
                valid.append(c)
        if self.jump == 'best':
            if bestcand is not None and not self.added_last:
                largs = self.spiro_args
                self.spiro_args = bestcand.framings[bestcand.creator]['args']
                self._log(logging.INFO,
                          "Jumped from {} to {}".format(largs, self.spiro_args))
        return valid

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
        self.spiro_args = best_artifact.framings[self.name]['args']
        best_artifact.in_domain = False
        best_artifact.self_criticism = 'reject'
        best_artifact.creation_time = self.age
        return best_artifact
