import aiomas

from agents.spiro.critic_test_agent import CriticTestAgent


class CriticOnlyAgent(CriticTestAgent):

    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)

    @aiomas.expose
    async def act(self):
        if len(self.stmem.artifacts) < self.stmem.length:
            await super().act()
        else:
            pass

    @aiomas.expose
    def ask_if_passes(self, artifact):
        self.opinion_asked_count += 1
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._novelty_threshold:
            artifact.add_eval(self, evaluation)
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            self.rejection_count += 1
            return False, artifact