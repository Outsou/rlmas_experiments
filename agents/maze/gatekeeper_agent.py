from agents.maze.maze_agent import MazeAgent

import aiomas


class GatekeeperAgent(MazeAgent):

    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.name = self.name + '_gk_N' + str(self.desired_novelty)
        self.subscriptions = []
        self.best_art = None
        self.published_creators = []

    @aiomas.expose
    def get_criticism(self, artifact, addr):
        self.subscriptions.append(addr)
        evaluation, _ = self.evaluate(artifact)

        if self.best_art == None or evaluation > self.best_art.evals[self.name]:
            self.best_art = artifact

        if evaluation >= self._novelty_threshold:
            self.learn(artifact)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    def get_published_creators(self):
        return self.published_creators

    @aiomas.expose
    async def publish(self):
        if self.best_art is None:
            return
        self.published_creators.append(self.best_art.creator)
        for sub in self.subscriptions:
            connection = await self.env.connect(sub)
            await connection.deliver_publication(self.best_art)
        self.best_art = None
        self.subscriptions = []


    @aiomas.expose
    async def act(self):
        pass
