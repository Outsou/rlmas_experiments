from artifacts.genetic_image_artifact import GeneticImageArtifact
from agents.generic.feature_agent import FeatureAgent
import matplotlib.pyplot as plt

import aiomas
import numpy as np
import logging
import time
import asyncio


class CollabAgent(FeatureAgent):

    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.connection = None
        self.collaborating = False
        self.has_turn = False
        self.artifact = None
        self.value_set = False
        self.create_kwargs['toolbox'].register(
            "evaluate", GeneticImageArtifact.evaluate, agent=self, shape=self.create_kwargs['shape'])
        self.best_received = None

    @aiomas.expose
    def init_connection(self, addr):
        self.connection = addr

    @aiomas.expose
    async def start_collab(self):
        self.value = np.random.random()
        self.value_set = True
        return self.value

    @aiomas.expose
    def pass_artifact(self, artifact, info=None):
        self.artifact = artifact
        eval, _ = self.evaluate(artifact)
        self._log(logging.INFO, 'Received artifact with value: ' + str(eval))
        if self.best_received is None or self.best_received.evals[self.name] < eval:
            artifact.add_eval(self, eval)
            self.best_received = artifact

    @aiomas.expose
    def get_artifact(self):
        return self.artifact

    @aiomas.expose
    def get_best_received(self):
        return self.best_received

    @aiomas.expose
    def get_eval(self):
        return self.evaluate(self.artifact)

    @aiomas.expose
    async def act(self):
        async def wait_for_collab():
            while not self.value_set:
                await asyncio.sleep(0.2)

        remote_agent = await self.env.connect(self.connection)
        if not self.collaborating:
            self._log(logging.INFO, 'Starting collab')
            remote_value = await remote_agent.start_collab()
            await wait_for_collab()
            self._log(logging.INFO, 'Started collab')
            self.collaborating = True
            if self.value < remote_value:
                self._log(logging.INFO, 'Starting first')
                self.artifact, _ = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
                while self.evaluate(self.artifact)[0] < 0.5:
                    self.artifact, _ = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
                self._log(logging.INFO, 'Initial value: ' + str(self.evaluate(self.artifact)[0]))
                await remote_agent.pass_artifact(self.artifact)
            else:
                self.has_turn = True
        elif self.has_turn:
            self._log(logging.INFO, 'Has turn')
            self.artifact = self.artifact_cls.work_on_artifact(self, self.artifact, self.create_kwargs, 1000)
            self.has_turn = False
            await remote_agent.pass_artifact(self.artifact)
            self.has_turn = False
        else:
            self.has_turn = True

