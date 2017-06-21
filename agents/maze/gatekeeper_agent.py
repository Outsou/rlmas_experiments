from agents.maze.maze_agent import MazeAgent

import aiomas


class GatekeeperAgent(MazeAgent):

    def __init__(self, environment, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.subscriptions = []
        self.best_art = None

    @aiomas.expose
    def get_criticism(self, artifact, addr):
        self.subscriptions.append(addr)
        evaluation, _ = self.evaluate(artifact)

        if self.best_art == None or evaluation > self.best_art.evals[self.name]:
            self.best_art = artifact

        if evaluation >= self._novelty_threshold:
            self.learn(artifact, 1)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    async def publish(self):
        for sub in self.subscriptions:
            connection = await self.env.connect(sub)
            await connection.deliver_publication(self.best_art)

    @aiomas.expose
    async def act(self):
        pass
