from agents.generic.feature_agent import FeatureAgent

import aiomas


class CriticAgent(FeatureAgent):
    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)

    @aiomas.expose
    async def act(self):
        pass
